import cv2
import tqdm
import clip
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from omegaconf import DictConfig
from typing import Any, Dict, List, Literal, Tuple, Union
from semantify.utils.general import get_plot_shape
from semantify.utils._3dmm_utils import ThreeDMMUtils
from semantify.utils.models_factory import ModelsFactory, Pytorch3dRenderer


class Image2ShapeUtils:
    def __init__(self):
        self._3dmm_utils = ThreeDMMUtils()

    @staticmethod
    def _gender_decider(arg: str) -> Literal["male", "female", "neutral"]:
        possible_gender = arg.split("smplx_")[1]
        assert possible_gender in ["male", "female", "neutral"], f"{possible_gender} is not supported"
        return possible_gender

    def _load_renderer(self, kwargs: Union[DictConfig, Dict[str, Any]]):
        self.renderer: Pytorch3dRenderer = ModelsFactory(self.model_type).get_renderer(**kwargs)

    def _load_body_pose(self, body_pose_path: str):
        self.body_pose: torch.Tensor = torch.from_numpy(np.load(body_pose_path))

    def _load_smplx_models(self, smplx_models_paths: Dict[str, str]):
        self.model: Dict[str, nn.Module] = {}
        self.labels: Dict[str, List[str]] = {}
        for model_name, model_path in smplx_models_paths.items():
            model, labels = self.utils.get_model_to_eval(model_path)
            labels = self._flatten_list_of_lists(labels)
            gender = self._gender_decider(model_name)
            self.model[gender] = model
            self.labels[gender] = labels

    def _load_flame_smal_models(self, model_path: str):
        self.model, labels = self.utils.get_model_to_eval(model_path)
        self.labels = self._flatten_list_of_lists(labels)

    def _load_comparison_data(self, path: Union[str, Path]):
        self.comparison_data: torch.Tensor = torch.from_numpy(np.load(path))

    def _load_images_generator(self):
        self.images_generator: List[Path] = sorted(list(self.data_dir.rglob(f"*.{self.suffix}")))

    def _load_clip_model(self):
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

    def _encode_labels(self):
        if isinstance(self.labels, dict):
            self.encoded_labels: Dict[str, torch.Tensor] = {
                gender: clip.tokenize(self.labels[gender]).to(self.device) for gender in self.labels.keys()
            }
        else:
            self.encoded_labels: torch.Tensor = clip.tokenize(self.labels).to(self.device)

    @staticmethod
    def _flatten_list_of_lists(list_of_lists: List[List[str]]) -> List[str]:
        return [item for sublist in list_of_lists for item in sublist]

    def _get_smplx_attributes(
        self,
        betas: torch.Tensor,
        gender: Literal["male", "female", "neutral"],
        get_smpl: bool = False,
        body_pose=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        betas = betas.cpu()
        if hasattr(self, "body_pose"):
            body_pose = self.body_pose
        verts, faces, vt, ft = self._3dmm_utils.get_smplx_model(
            betas=betas, gender=gender, body_pose=body_pose, get_smpl=get_smpl
        )
        if get_smpl:
            verts += self._3dmm_utils.smpl_offset_numpy
        else:
            verts += self._3dmm_utils.smplx_offset_numpy
        return verts, faces, vt, ft

    def _get_flame_attributes(
        self, pred_vec: torch.Tensor, gender: Literal["male", "female", "neutral"]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.with_face:
            flame_out = self._3dmm_utils.get_flame_model(expression_params=pred_vec.cpu(), gender=gender)
        else:
            flame_out = self._3dmm_utils.get_flame_model(shape_params=pred_vec.cpu(), gender=gender)
        return flame_out

    def _get_smal_attributes(self, pred_vec: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, None, None]:
        smal_out = self._3dmm_utils.get_smal_model(beta=pred_vec.cpu())
        return smal_out

    def get_render_mesh_kwargs(
        self, pred_vec: torch.Tensor, gender: Literal["male", "female", "neutral"], get_smpl: bool = False
    ) -> Dict[str, np.ndarray]:
        if self.model_type == "smplx" or self.model_type == "smpl":
            out = self._get_smplx_attributes(pred_vec=pred_vec, gender=gender, get_smpl=get_smpl)
        elif self.model_type == "flame":
            out = self._get_flame_attributes(pred_vec=pred_vec)
        elif self.model_type == "smal":
            out = self._get_smal_attributes(pred_vec=pred_vec)

        kwargs = {"verts": out[0], "faces": out[1], "vt": out[2], "ft": out[3]}

        return kwargs

    @staticmethod
    def adjust_rendered_img(img: torch.Tensor) -> np.ndarray:
        img = np.clip(img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)
        return img

    def create_video_from_dir(self, dir_path: Union[str, Path], image_shape: Tuple[int, int]):
        dir_path = Path(dir_path)
        out_vid_path = dir_path.parent / "out_vid.mp4"
        out_vid = cv2.VideoWriter(out_vid_path.as_posix(), cv2.VideoWriter_fourcc(*"mp4v"), 30, image_shape[::-1])
        sorted_frames = sorted(dir_path.iterdir(), key=lambda x: int(x.stem))
        for frame in tqdm(sorted_frames, desc="Creating video", total=len(sorted_frames)):
            out_vid.write(cv2.imread(frame.as_posix()))
        out_vid.release()

    def _save_images_collage(self, images: List[np.ndarray]):
        collage_shape = get_plot_shape(len(images))[0]
        images_collage = []
        for i in range(collage_shape[0]):
            images_collage.append(np.hstack(images[i * collage_shape[1] : (i + 1) * collage_shape[1]]))
        images_collage = np.vstack([image for image in images_collage])
        cv2.imwrite(self.images_out_path.as_posix(), images_collage)
        self.num_img += 1
        self.images_out_path = self.images_out_path.parent / f"{self.num_img}.png"

    @staticmethod
    def mesh_attributes_to_kwargs(
        attributes: Union[
            Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ],
        to_tensor: bool = False,
    ) -> Union[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        if isinstance(attributes, dict):
            kwargs = {}
            for key, value in attributes.items():
                kwargs[key] = {"verts": value[0], "faces": value[1], "vt": value[2], "ft": value[3]}
                if to_tensor:
                    kwargs[key] = {k: torch.tensor(v)[None] for k, v in kwargs[key].items()}
        else:
            kwargs = {"verts": attributes[0], "faces": attributes[1], "vt": attributes[2], "ft": attributes[3]}
            if to_tensor:
                kwargs = {k: torch.tensor(v)[None] for k, v in kwargs.items()}
        return kwargs
