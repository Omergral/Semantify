import cv2
import json
import clip
import torch
import tkinter
import hydra
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import ImageTk, Image
from pytorch3d.io import save_obj
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Literal, List, Union
from semantify.SPIN.spin_model import hmr, process_image
from semantify.utils import Utils, ModelsFactory, append_to_root_dir, get_model_abs_path

OmegaConf.register_new_resolver("get_model_abs_path", get_model_abs_path)


class SlidersApp:
    def __init__(
        self,
        device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        texture: str = None,
        num_coeffs: int = 10,
        on_parameters: bool = False,
        model_type: Literal["smplx", "flame", "smpl", "smal"] = None,
        out_dir: str = None,
        gender: Literal["male", "female", "neutral"] = "neutral",
        predict_jaw_pose: bool = False,
        renderer_kwargs: DictConfig = None,
        sliders_limiters: DictConfig = None,
        model_path: str = None,
        image_path: Union[str, Literal["random"]] = None,
        image2shape: bool = False,
        comparison_mode: bool = False,
        A_pose: bool = False,
        show_values: bool = False,
        hbw_gt: str = None,
        specific: Literal["expression", "shape"] = None,
        with_face: bool = False,
    ):

        self.root = None
        self.img_label = None
        self.vertex_colors = None
        self.hbw_gt = hbw_gt
        self.predicted_coeffs = None
        self.visualize_error = False
        self.device = device
        self.texture = texture
        self.A_pose = A_pose
        self.image2shape = image2shape
        self.show_values = show_values
        self.num_coeffs = num_coeffs
        self.on_parameters = on_parameters
        self.predict_jaw_pose = predict_jaw_pose
        self.comparison_mode = comparison_mode

        self.model_type = model_type
        if sliders_limiters is not None:
            if model_type in ["smplx", "smpl"]:
                self.sliders_limiters = sliders_limiters[gender]
            elif model_type == "flame":
                self.sliders_limiters = sliders_limiters["expression" if with_face else "shape"]
            elif model_type == "smal":
                self.sliders_limiters = sliders_limiters
            else:
                raise ValueError(f"model_type: {model_type} is not supported")
        else:
            self.sliders_limiters = None

        if image_path == "random":
            # FOR USER STUDY ONLY - DO NOT USE
            images_dir = (
                f"/home/nadav2/dev/data/CLIP2Shape/outs/sliders_vs_blendshapes/images_to_fit/{model_type}/{gender}"
            )
            image_path = np.random.choice(list(Path(images_dir).rglob("*.png"))).as_posix()
            print(f"image_path: {image_path}")

        self._assertions(image_path=image_path, model_path=model_path, model_type=model_type, gender=gender)

        self.outpath = None
        if out_dir is not None:
            if not Path(out_dir).exists():
                Path(out_dir).mkdir(parents=True)
            try:
                img_id = int(sorted(list(Path(out_dir).glob("*.png")), key=lambda x: int(x.stem))[-1].stem) + 1
            except IndexError or ValueError:
                img_id = 0
            self.outpath = Path(out_dir) / f"{img_id}.png"

        self.params = []
        self.utils = Utils()
        self.models_factory = ModelsFactory(self.model_type)
        self.gender = gender

        if self.on_parameters:
            self.with_face = with_face
        else:
            self.with_face = True if model_type == "flame" and specific == "expression" else False

        if self.A_pose:
            self.body_pose = self.utils.body_pose.clone()
        else:
            self.body_pose = None

        if self.comparison_mode:
            self.color_map = plt.get_cmap("coolwarm")
            self._get_target_shape(image_path.replace(".png", ".json"))

        if image_path is not None:
            self.target_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            if self.model_type in ["smpl", "smplx"] and not on_parameters:
                spin_model = hmr(append_to_root_dir("SPIN/smpl_mean_params.npz")).to(device)
                checkpoint = torch.load(append_to_root_dir("SPIN/model_checkpoint.pt"))
                spin_model.load_state_dict(checkpoint["model"], strict=False)
                spin_model.eval()
                img, img_norm = process_image(image_path)
                with torch.no_grad():
                    out = spin_model(img_norm.to(device))
                self.body_pose = out[0][:, 1:-2].cpu()

        self.model_kwargs = self.models_factory.get_default_params(with_face, num_coeffs=num_coeffs)
        if self.model_type == "smpl":
            self.model_kwargs["get_smpl"] = True
        if hasattr(self, "num_coeffs"):
            self.model_kwargs["num_coeffs"] = self.num_coeffs
        if hasattr(self, "gender"):
            self.model_kwargs["gender"] = self.gender
        if hasattr(self, "body_pose") and self.model_type in ["smpl", "smplx"]:
            self.model_kwargs["body_pose"] = self.body_pose

        self.verts, self.faces, self.vt, self.ft = self.models_factory.get_model(**self.model_kwargs)
        if self.model_type == "smplx":
            self.verts += self.utils.smplx_offset_numpy  # center the model with offsets
        if self.model_type == "smpl":
            self.verts += self.utils.smpl_offset_numpy

        self.renderer_kwargs = renderer_kwargs
        self.renderer = self.models_factory.get_renderer(**self.renderer_kwargs)

        self.dist = renderer_kwargs["dist"]
        self.azim = 0.0
        self.elev = 0.0

        img = self.renderer.render_mesh(verts=self.verts, faces=self.faces[None], vt=self.vt, ft=self.ft)
        self.img = self.adjust_rendered_img(img)

        self.production_scales = []
        self.camera_scales = {}
        self.initialize_params()

        self.ignore_random_jaw = True if model_type == "flame" and with_face else False

        if model_path is not None:
            self.model, labels = self.utils.get_model_to_eval(model_path)
            self._get_sliders_values(labels, image_path)
            self.input_for_model = torch.tensor(list(self.sliders_values.values()), dtype=torch.float32)[None]

    def initialize_params(self):
        if self.on_parameters:
            if self.model_type == "smplx" or self.model_type == "smpl":
                self.betas = self.model_kwargs["betas"]
                self.expression = self.model_kwargs["expression"]
                self.params.append(self.betas)
                self.params.append(self.expression)

            elif self.model_type == "flame":
                if self.with_face:
                    self.face_expression = self.model_kwargs["expression_params"][..., :10]
                    self.params.append(self.face_expression)
                else:
                    self.face_shape = self.model_kwargs["shape_params"][..., :10]
                    self.params.append(self.face_shape)

            else:
                self.beta = self.model_kwargs["beta"]
                self.params.append(self.beta)

    def _assertions(self, image_path: str = None, model_path: str = None, model_type: str = None, gender: str = None):
        if image_path is not None:
            assert Path(image_path).exists(), "Image path should be a valid path"
            assert (
                model_path is not None or self.comparison_mode is True
            ), "Model path should be a valid path if image path is provided"
        assert model_type in ["smplx", "flame", "smal", "smpl"], "Model type should be smplx, smpl, flame or smal"
        assert gender in ["male", "female", "neutral"], "Gender is not valid"

    def _get_target_shape(self, json_path: str):
        get_smpl = True if self.model_type == "smpl" else False
        if self.hbw_gt is not None:
            verts = np.load(self.hbw_gt)
            _, faces, vt, ft = self.models_factory.get_model()
        else:
            feature = self.name
            with open(json_path) as f:
                data = json.load(f)
            target_shape_list = data[feature]
            target_shape_tensor = torch.tensor(target_shape_list, dtype=torch.float32)
            verts, faces, vt, ft = self.models_factory.get_model(
                **{
                    feature: target_shape_tensor,
                    "get_smpl": get_smpl,
                    "body_pose": self.body_pose,
                    "gender": self.gender,
                }
            )
        if get_smpl:
            verts += self.utils.smpl_offset_numpy
        else:
            verts += self.utils.smplx_offset_numpy
        self.target_mesh_features = {
            "verts": verts,
            "faces": faces,
            "vt": vt,
            "ft": ft,
        }
        # self.target_coeffs = target_shape_tensor

    @staticmethod
    def _flatten_list_of_lists(list_of_lists: List[List[str]]) -> List[str]:
        return [item for sublist in list_of_lists for item in sublist]

    def _get_sliders_values(self, labels: List[List[str]], image_path: str = None):
        if image_path is None:  # or self.comparison_mode:
            self.sliders_values = {label[0]: 20 for label in labels}
        else:
            if not hasattr(self, "clip_model"):
                self._load_clip_model()
            enc_image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            labels = self._flatten_list_of_lists(labels)
            enc_labels = clip.tokenize(labels).to(self.device)
            with torch.no_grad():
                clip_scores = self.clip_model(enc_image, enc_labels)[0].float()
                self.sliders_values = {label: clip_scores[0, i].item() for i, label in enumerate(labels)}

    def _load_clip_model(self):
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

    def update_betas(self, idx: int):
        def update_betas_values(value: float):
            if isinstance(value, str):
                value = float(value)
            self.betas[0, idx] = value
            self.predicted_coeffs = self.betas.clone()
            get_smpl = True if self.model_type == "smpl" else False
            self.verts, self.faces, self.vt, self.ft = self.utils.get_smplx_model(
                betas=self.betas,
                expression=self.expression,
                gender=self.gender,
                get_smpl=get_smpl,
                body_pose=self.body_pose,
            )
            if self.model_type == "smplx":
                self.verts += self.utils.smplx_offset_numpy
            if self.model_type == "smpl":
                self.verts += self.utils.smpl_offset_numpy
            img = self.renderer.render_mesh(
                verts=self.verts,
                faces=self.faces[None],
                vt=self.vt,
                ft=self.ft,
                rotate_mesh=[{"degrees": self.azim, "axis": "y"}, {"degrees": self.elev, "axis": "x"}],
            )
            img = self.adjust_rendered_img(img)
            self.img = img
            img = ImageTk.PhotoImage(image=img)
            self.img_label.configure(image=img)
            self.img_label.image = img

        return update_betas_values

    def update_face_shape(self, idx: int):
        def update_face_shape_values(value: float):
            if isinstance(value, str):
                value = float(value)
            self.face_shape[0, idx] = value

            self.verts, self.faces, self.vt, self.ft = self.utils.get_flame_model(shape_params=self.face_shape)

            img = self.renderer.render_mesh(verts=self.verts, faces=self.faces[None], vt=self.vt, ft=self.ft)
            img = self.adjust_rendered_img(img)
            self.img = img
            img = ImageTk.PhotoImage(image=img)
            self.img_label.configure(image=img)
            self.img_label.image = img

        return update_face_shape_values

    def update_face_expression(self, idx: int):
        def update_face_expression_values(value: float):
            if isinstance(value, str):
                value = float(value)
            self.face_expression[0, idx] = value

            self.verts, self.faces, self.vt, self.ft = self.utils.get_flame_model(
                expression_params=self.face_expression[..., :10]
            )
            img = self.renderer.render_mesh(verts=self.verts, faces=self.faces[None], vt=self.vt, ft=self.ft)
            img = self.adjust_rendered_img(img)
            self.img = img
            img = ImageTk.PhotoImage(image=img)
            self.img_label.configure(image=img)
            self.img_label.image = img

        return update_face_expression_values

    def update_beta_shape(self, idx: int):
        def update_beta_shape_values(value: float):
            if isinstance(value, str):
                value = float(value)
            self.beta[0, idx] = value
            self.verts, self.faces, self.vt, self.ft = self.utils.get_smal_model(beta=self.beta)
            img = self.renderer.render_mesh(verts=self.verts, faces=self.faces[None], vt=self.vt, ft=self.ft)
            img = self.adjust_rendered_img(img)
            self.img = img
            img = ImageTk.PhotoImage(image=img)
            self.img_label.configure(image=img)
            self.img_label.image = img

        return update_beta_shape_values

    def update_labels(self, idx: int):
        def update_labels_values(value: float):
            if isinstance(value, str):
                value = float(value)
            self.input_for_model[0, idx] = value
            # print(self.input_for_model)  # for debug
            with torch.no_grad():
                out = self.model(self.input_for_model.to(self.device)).cpu()
                self.predicted_coeffs = out.clone()
                if self.model_type == "smplx" or self.model_type == "smpl":
                    betas = out
                    get_smpl = True if self.model_type == "smpl" else False
                    self.verts, self.faces, self.vt, self.ft = self.utils.get_smplx_model(
                        betas=betas,
                        gender=self.gender,
                        num_coeffs=self.num_coeffs,
                        get_smpl=get_smpl,
                        body_pose=self.body_pose,
                    )
                    if get_smpl:
                        self.verts += self.utils.smpl_offset_numpy
                    else:
                        self.verts += self.utils.smplx_offset_numpy
                elif self.model_type == "flame":
                    if self.with_face:
                        if self.predict_jaw_pose:
                            jaw_pose = out[..., 0]
                            expression_params = out[..., 1:]
                            self.verts, self.faces, self.vt, self.ft = self.utils.get_flame_model(
                                expression_params=expression_params, jaw_pose=jaw_pose
                            )
                        else:
                            expression_params = out
                            self.verts, self.faces, self.vt, self.ft = self.utils.get_flame_model(
                                expression_params=expression_params
                            )
                    else:
                        self.verts, self.faces, self.vt, self.ft = self.utils.get_flame_model(shape_params=out)

                else:
                    self.verts, self.faces, self.vt, self.ft = self.utils.get_smal_model(beta=out)

            img = self.renderer.render_mesh(
                verts=self.verts,
                faces=self.faces[None],
                vt=self.vt,
                ft=self.ft,
                rotate_mesh=[{"degrees": self.azim, "axis": "y"}, {"degrees": self.elev, "axis": "x"}],
                texture_color_values=self.vertex_colors,
            )
            img = self.adjust_rendered_img(img)
            self.img = img
            img = ImageTk.PhotoImage(image=img)
            self.img_label.configure(image=img)
            self.img_label.image = img

        return update_labels_values

    def adjust_rendered_img(self, img: torch.Tensor):
        img = np.clip(img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)
        img = cv2.resize(img, (512, 512))
        if self.comparison_mode:
            if self.visualize_error:
                diffs = self._calc_vertices_distance()
                # vertex_colors = self.color_map(diffs / diffs.max())[:, :3]
                vertex_colors = self.color_map(diffs / 0.1094)[:, :3]
                vertex_colors = torch.tensor(vertex_colors).float().to(self.device)
                if vertex_colors.ndim == 2:
                    vertex_colors = vertex_colors[None, ...]
            else:
                vertex_colors = torch.ones(*self.verts[None].shape, device=self.device) * torch.tensor(
                    [0.7, 0.7, 0.7], device=self.device
                )

            self.vertex_colors = vertex_colors
            self.target_mesh_features.update({"texture_color_values": vertex_colors})

            if hasattr(self, "azim"):
                self.target_mesh_features["rotate_mesh"] = [
                    {"degrees": self.azim, "axis": "y"},
                    {"degrees": self.elev, "axis": "x"},
                ]
                target_img = self.renderer.render_mesh(**self.target_mesh_features)
                target_img = np.clip(target_img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)
                self.target_image = cv2.resize(target_img, (512, 512))

            img = self.renderer.render_mesh(
                verts=self.verts,
                faces=self.faces[None],
                vt=self.vt,
                ft=self.ft,
                rotate_mesh=[{"degrees": self.azim, "axis": "y"}, {"degrees": self.elev, "axis": "x"}],
                texture_color_values=self.vertex_colors,
            )
            img = np.clip(img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)
            img = cv2.resize(img, (512, 512))

        if hasattr(self, "target_image"):
            target_image = cv2.resize(self.target_image, (512, 512))
            img = np.concatenate([img, target_image], axis=1)

        if self.visualize_error:
            img = self.write_diff_on_img(img, np.sqrt(diffs).mean())
        return Image.fromarray(img)

    def _calc_coeffs_distance(self) -> float:
        distance = np.linalg.norm(self.predicted_coeffs - self.target_coeffs)
        return distance

    def _calc_vertices_distance(self):
        distance = np.linalg.norm(self.verts - self.target_mesh_features["verts"], axis=-1)
        return distance

    def write_diff_on_img(self, img: np.ndarray, distance: float = None):
        if distance is None:
            distance = self._calc_coeffs_distance()
        x_coord = 10 if self.image2shape else 320
        cv2.putText(
            img,
            f"verts_diff: {distance:.4f}",
            (x_coord, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return img

    def add_texture(self):
        if self.texture is not None:
            self.renderer_kwargs.update({"use_tex": True})
            self.renderer = self.models_factory.get_renderer(**self.renderer_kwargs)
            img = self.renderer.render_mesh(verts=self.verts, faces=self.faces[None], vt=self.vt, ft=self.ft)
            img = self.adjust_rendered_img(img)
            self.img = img
            img = ImageTk.PhotoImage(image=img)
            self.img_label.configure(image=img)
            self.img_label.image = img

    def remove_texture(self):
        self.renderer_kwargs.update({"use_tex": False})
        self.renderer = self.models_factory.get_renderer(**self.renderer_kwargs)
        img = self.renderer.render_mesh(
            verts=self.verts,
            faces=self.faces[None],
            vt=self.vt,
            ft=self.ft,
            rotate_mesh=[{"degrees": self.azim, "axis": "y"}, {"degrees": self.elev, "axis": "x"}],
        )
        img = self.adjust_rendered_img(img)
        self.img = img
        img = ImageTk.PhotoImage(image=img)
        self.img_label.configure(image=img)
        self.img_label.image = img

    def update_camera_zoom(self, value: float):
        if isinstance(value, str):
            value = float(value)
        self.renderer_kwargs.update({"dist": value})
        self.renderer = self.models_factory.get_renderer(**self.renderer_kwargs)
        img = self.renderer.render_mesh(
            verts=self.verts,
            faces=self.faces[None],
            vt=self.vt,
            ft=self.ft,
            rotate_mesh=[{"degrees": self.azim, "axis": "y"}, {"degrees": self.elev, "axis": "x"}],
        )
        img = self.adjust_rendered_img(img)
        self.img = img
        img = ImageTk.PhotoImage(image=img)
        self.img_label.configure(image=img)
        self.img_label.image = img

    def update_camera_azim(self, value: float):
        if isinstance(value, str):
            value = float(value)
        self.azim = value
        img = self.renderer.render_mesh(
            verts=self.verts,
            faces=self.faces[None],
            vt=self.vt,
            ft=self.ft,
            rotate_mesh=[{"degrees": value, "axis": "y"}, {"degrees": self.elev, "axis": "x"}],
        )
        img = self.adjust_rendered_img(img)
        self.img = img
        img = ImageTk.PhotoImage(image=img)
        self.img_label.configure(image=img)
        self.img_label.image = img

    def update_camera_elev(self, value: float):
        if isinstance(value, str):
            value = float(value)
        img = self.renderer.render_mesh(
            verts=self.verts,
            faces=self.faces[None],
            vt=self.vt,
            ft=self.ft,
            rotate_mesh=[{"degrees": self.azim, "axis": "y"}, {"degrees": value, "axis": "x"}],
        )
        img = self.adjust_rendered_img(img)
        self.elev = value
        self.img = img
        img = ImageTk.PhotoImage(image=img)
        self.img_label.configure(image=img)
        self.img_label.image = img

    @property
    def name(self):
        if self.model_type in ["smplx", "smpl"]:
            return "betas"
        elif self.model_type == "flame":
            if self.with_face:
                return "expression_params"
            else:
                return "shape_params"
        elif self.model_type == "smal":
            return "beta"
        else:
            raise ValueError("Unknown model type")

    def mouse_wheel(self, action):
        """
        control the scrollbar using the mousewheel
        """
        scroll = -1 if action.delta > 0 else 1
        self.parameters_canvas.yview(scroll, "units")

    def save_png(self):
        resize = False if self.comparison_mode else True
        if self.on_parameters:
            self.img = np.array(self.img)
            self.renderer.save_rendered_image(self.img, self.outpath.as_posix(), resize=resize)
            key = self.name
            params = {key: [self.params[0].tolist()[0]]}
            with open(self.outpath.with_suffix(".json"), "w") as f:
                json.dump(params, f)
        else:
            self.img = np.array(self.img)
            # self.renderer.save_rendered_image(self.img, self.outpath.as_posix(), resize=resize)
            self.renderer.save_rendered_image(self.img, self.outpath.as_posix(), resize=False)
        new_img_id = int(self.outpath.stem) + 1
        self.outpath = self.outpath.parent / f"{new_img_id}.png"

    def save_obj(self):
        if self.outpath is not None:
            if self.outpath.suffix == ".obj":
                obj_path = self.outpath
            elif self.outpath.suffix == ".png":
                obj_path = self.outpath.with_suffix(".obj")
        else:
            obj_path = "./out.obj"
        obj_path = str(obj_path)
        save_obj(
            f=obj_path,
            verts=torch.tensor(self.verts).squeeze(),
            faces=torch.tensor(self.faces).squeeze(),
        )
        if self.outpath is not None:
            new_img_id = int(self.outpath.stem) + 1
            self.outpath = self.outpath.parent / f"{new_img_id}.png"

    def random_button(self):
        if self.on_parameters:
            random_params = self.models_factory.get_random_params(self.with_face)[self.name][0, :10]
            scales_list = self.production_scales if not self.ignore_random_jaw else self.production_scales[:-1]
            for idx, scale in enumerate(scales_list):
                scale.set(random_params[idx].item())

    def update_expression(self, idx: int):
        def update_expression_values(value: float):
            if isinstance(value, str):
                value = float(value)
            self.expression[0, idx] = value

            self.verts, self.faces, self.vt, self.ft = self.utils.get_smplx_model(
                betas=self.betas, body_pose=self.body_pose, expression=self.expression
            )
            self.renderer.render_mesh(
                verts=self.verts,
                faces=self.faces[None],
                vt=self.vt,
                ft=self.ft,
                rotate_mesh=[{"degrees": self.azim, "axis": "y"}, {"degrees": self.elev, "axis": "x"}],
            )

        return update_expression_values

    def reset_parameters(self):
        if self.on_parameters:
            for scale in self.production_scales:
                scale.set(0)
        else:
            for label in self.production_scales:
                if self.sliders_limiters is not None:
                    label.set(self.sliders_limiters[label.cget("label").replace("- ", "").replace(" +", "")][-1])
                else:
                    label.set(20)

    def reset_pose(self):
        if self.body_pose is None and hasattr(self, "prev_pose"):
            self.body_pose = self.prev_pose
        else:
            self.prev_pose = self.body_pose.clone()
            self.body_pose = None
        with torch.no_grad():
            out = self.model(self.input_for_model.to(self.device)).cpu()
        self.predicted_coeffs = out.clone()
        get_smpl = True if self.model_type == "smpl" else False
        self.verts, self.faces, self.vt, self.ft = self.utils.get_smplx_model(
            betas=out,
            gender=self.gender,
            num_coeffs=self.num_coeffs,
            get_smpl=get_smpl,
            body_pose=self.body_pose,
        )
        if get_smpl:
            self.verts += self.utils.smpl_offset_numpy
        else:
            self.verts += self.utils.smplx_offset_numpy
        img = self.renderer.render_mesh(
            verts=self.verts,
            faces=self.faces[None],
            vt=self.vt,
            ft=self.ft,
            rotate_mesh=[{"degrees": self.azim, "axis": "y"}, {"degrees": self.elev, "axis": "x"}],
            texture_color_values=self.vertex_colors,
        )
        img = self.adjust_rendered_img(img)
        self.img = img
        img = ImageTk.PhotoImage(image=img)
        self.img_label.configure(image=img)
        self.img_label.image = img

    def visual_error(self):
        if self.visualize_error:
            self.visualize_error = False
        else:
            self.visualize_error = True
        img = self.renderer.render_mesh(
            verts=self.verts,
            faces=self.faces[None],
            vt=self.vt,
            ft=self.ft,
            rotate_mesh=[{"degrees": self.azim, "axis": "y"}, {"degrees": self.elev, "axis": "x"}],
        )
        img = self.adjust_rendered_img(img)
        self.img = img
        img = ImageTk.PhotoImage(image=img)
        self.img_label.configure(image=img)
        self.img_label.image = img

    def reset_cam_params(self):
        self.renderer_kwargs.update({"dist": self.dist})
        self.azim = 0
        self.elev = 0
        self.renderer = self.models_factory.get_renderer(**self.renderer_kwargs)
        img = self.renderer.render_mesh(
            verts=self.verts,
            faces=self.faces[None],
            vt=self.vt,
            ft=self.ft,
            rotate_mesh=[{"degrees": self.azim, "axis": "y"}, {"degrees": self.elev, "axis": "x"}],
        )
        img = self.adjust_rendered_img(img)
        self.img = img
        img = ImageTk.PhotoImage(image=img)
        self.img_label.configure(image=img)
        self.img_label.image = img
        self.camera_scales["azim"].set(self.azim)
        self.camera_scales["elev"].set(self.elev)
        self.camera_scales["dist"].set(self.dist)

    def _zeros_to_concat(self):
        if self.model_type == "smplx":
            pass
        elif self.model_type == "flame":
            if self.with_face:
                return torch.zeros((40)).tolist()
            else:
                return torch.zeros(1, 90).tolist()
        else:
            return torch.zeros((31)).tolist()

    def create_application(self):

        # ------------------ Create the root window ------------------
        self.root = tkinter.Tk()
        self.root.title("Semantify Sliders Application - PyTorch3D")
        if self.comparison_mode:
            self.root.geometry("2000x2000")
        else:
            if self.on_parameters:
                self.root.geometry("1000x1000")
            else:
                self.root.geometry("800x560")

        img_coords = (80, 10)

        parameters_main_frame = tkinter.Frame(self.root, bg="white", borderwidth=0)
        image_main_frame = tkinter.Frame(self.root, bg="white", borderwidth=0)

        img_frame = tkinter.Frame(
            self.root, highlightbackground="white", highlightthickness=0, bg="white", borderwidth=0
        )
        parameters_frame = tkinter.Frame(
            self.root, highlightbackground="white", highlightthickness=0, bg="white", borderwidth=0
        )
        parameters_main_frame.pack(fill=tkinter.BOTH, expand=True, side=tkinter.LEFT)
        image_main_frame.pack(fill=tkinter.BOTH, expand=True, side=tkinter.RIGHT)
        # ------------------------------------------------------------

        # ------------------------ Image ----------------------------
        img_canvas = tkinter.Canvas(image_main_frame, bg="white", highlightbackground="white", borderwidth=0)
        img_canvas.pack(fill=tkinter.BOTH, expand=True, side=tkinter.LEFT)
        img_canvas.create_window(img_coords, window=img_frame, anchor=tkinter.N)
        img = ImageTk.PhotoImage(image=self.img)
        self.img_label = tkinter.Label(img_frame, image=img, borderwidth=0)
        self.img_label.pack(fill=tkinter.BOTH, expand=True, side=tkinter.LEFT)
        # ------------------------------------------------------------

        # ----------------------- Parameters -------------------------
        self.parameters_canvas = tkinter.Canvas(
            parameters_main_frame, bg="white", highlightbackground="white", borderwidth=0
        )
        self.parameters_canvas.pack(side=tkinter.LEFT, padx=0, pady=0, anchor="nw")
        self.parameters_canvas.create_window((0, 0), window=parameters_frame, anchor=tkinter.NW)
        # ------------------------------------------------------------

        # ------------------- Parameters Scale Bars ------------------
        if self.on_parameters:

            if self.model_type == "smplx" or self.model_type == "smpl":
                scale_kwargs = self.get_parameters_scale_kwargs()
                for beta in range(self.betas.shape[1]):
                    betas_scale = tkinter.Scale(
                        parameters_frame,
                        label=f"beta {beta}",
                        command=self.update_betas(beta),
                        **scale_kwargs,
                    )
                    self.production_scales.append(betas_scale)
                    betas_scale.set(0)
                    betas_scale.pack()

            elif self.model_type == "flame":
                scale_kwargs = self.get_parameters_scale_kwargs()
                if self.with_face:
                    for label in range(self.face_expression.shape[1]):
                        label_tag = (
                            f"- expression {label} +" if label != self.face_expression.shape[1] - 1 else "jaw pose"
                        )
                        face_expression_scale = tkinter.Scale(
                            parameters_frame,
                            label=label_tag,
                            command=self.update_face_expression(label),
                            **scale_kwargs,
                        )
                        self.production_scales.append(face_expression_scale)
                        face_expression_scale.set(0)
                        face_expression_scale.pack()
                else:
                    scale_kwargs = self.get_parameters_scale_kwargs()
                    for label in range(self.face_shape.shape[1]):
                        face_shape_scale = tkinter.Scale(
                            parameters_frame,
                            label=f"- shape param {label} +",
                            command=self.update_face_shape(label),
                            **scale_kwargs,
                        )
                        self.production_scales.append(face_shape_scale)
                        face_shape_scale.set(0)
                        face_shape_scale.pack()

            else:
                scale_kwargs = self.get_smal_scale_kwargs()
                for label in range(self.beta.shape[1]):
                    beta_shape_scale = tkinter.Scale(
                        parameters_frame,
                        label=f"- beta param {label} +",
                        command=self.update_beta_shape(label),
                        **scale_kwargs,
                    )
                    self.production_scales.append(beta_shape_scale)
                    beta_shape_scale.set(0)
                    beta_shape_scale.pack()

        else:
            scale_kwargs = self.get_stats_scale_kwargs()
            for idx, label in enumerate(self.sliders_values.keys()):
                if self.sliders_limiters is not None and self.target_image is None:
                    scale_kwargs["from_"] = self.sliders_limiters[label][0]
                    scale_kwargs["to"] = self.sliders_limiters[label][1]
                    self.sliders_values[label] = self.sliders_limiters[label][-1]
                label_scale = tkinter.Scale(
                    parameters_frame,
                    label=f"- {label} +",
                    command=self.update_labels(idx),
                    **scale_kwargs,
                )
                self.production_scales.append(label_scale)
                label_scale.set(self.sliders_values[label])
                label_scale.pack()
        # ------------------------------------------------------------

        # --------------------- Camera Controls ----------------------
        zoom_scale_kwarg = self.get_zoom_scale_kwargs()
        zoom_in_scale = tkinter.Scale(
            parameters_frame,
            label="Zoom - in <-> out",
            command=lambda x: self.update_camera_zoom(x),
            **zoom_scale_kwarg,
        )
        zoom_in_scale.set(self.dist)
        zoom_in_scale.pack(pady=(50, 0))
        self.camera_scales["dist"] = zoom_in_scale

        azim_scale_kwarg = self.get_azim_scale_kwargs()
        azim_scale = tkinter.Scale(
            parameters_frame,
            label="azim - left <-> right",
            command=lambda x: self.update_camera_azim(x),
            **azim_scale_kwarg,
        )
        azim_scale.set(self.azim)
        azim_scale.pack()
        self.camera_scales["azim"] = azim_scale

        elev_scale_kwarg = self.get_elev_scale_kwargs()
        elev_scale = tkinter.Scale(
            parameters_frame,
            label="elev - down <-> up",
            command=lambda x: self.update_camera_elev(x),
            **elev_scale_kwarg,
        )
        elev_scale.set(self.elev)
        elev_scale.pack()
        self.camera_scales["elev"] = elev_scale
        # ------------------------------------------------------------

        # ------------------------ Scroll Bar ------------------------
        # adding Scrollbar
        y_scroll = tkinter.Scrollbar(
            img_frame,
            orient=tkinter.VERTICAL,
            command=self.parameters_canvas.yview,
            bg="white",
            troughcolor="white",
            activebackground="black",
            highlightbackground="white",
            width=8,
        )
        parameters_main_frame.bind(
            "<Configure>", lambda e: self.parameters_canvas.configure(scrollregion=self.parameters_canvas.bbox("all"))
        )
        self.parameters_canvas.configure(yscrollcommand=y_scroll.set)
        y_scroll.pack(
            fill=tkinter.Y,
            side=tkinter.LEFT,
        )
        # ------------------------------------------------------------

        # ------------------------ Buttons --------------------------
        reset_button_kwargs = self.get_reset_button_kwargs()

        # all reset button
        reset_all_button = tkinter.Button(
            parameters_frame, text="Reset Parameters", command=lambda: self.reset_parameters(), **reset_button_kwargs
        )
        reset_all_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP, pady=(50, 0))

        # reset camera button
        reset_camera_button = tkinter.Button(
            parameters_frame, text="Reset Camera", command=lambda: self.reset_cam_params(), **reset_button_kwargs
        )
        reset_camera_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)

        # add texture button
        add_texture_button = tkinter.Button(
            parameters_frame, text="Add Texture", command=lambda: self.add_texture(), **reset_button_kwargs
        )
        add_texture_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)

        # remove texture button
        remove_texture_button = tkinter.Button(
            parameters_frame, text="Remove Texture", command=lambda: self.remove_texture(), **reset_button_kwargs
        )
        remove_texture_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)

        # save image & params button
        save_img_n_params_button = tkinter.Button(
            parameters_frame, text="save png", command=lambda: self.save_png(), **reset_button_kwargs
        )
        save_img_n_params_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)

        # save image & params button
        save_obj_button = tkinter.Button(
            parameters_frame, text="save obj", command=lambda: self.save_obj(), **reset_button_kwargs
        )
        save_obj_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)

        # generate random params button
        random_params_button = tkinter.Button(
            parameters_frame, text="random params", command=lambda: self.random_button(), **reset_button_kwargs
        )
        random_params_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)

        # reset pose button - only for smpl and smplx
        if self.model_type in ["smpl", "smplx"] and hasattr(self, "target_image"):
            reset_pose_button = tkinter.Button(
                parameters_frame, text="reset pose", command=lambda: self.reset_pose(), **reset_button_kwargs
            )
            reset_pose_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)

        # if in comparison_mode, add option to visualize the error
        if self.comparison_mode:
            visual_error_button = tkinter.Button(
                parameters_frame, text="error", command=lambda: self.visual_error(), **reset_button_kwargs
            )
            visual_error_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)
        # ------------------------------------------------------------
        self.parameters_canvas.bind("<MouseWheel>", self.mouse_wheel)
        self.root.mainloop()

    def get_zoom_scale_kwargs(self) -> Dict[str, Any]:
        if self.model_type == "flame":
            return {
                "from_": 0.45,
                "to": 1,
                "resolution": 0.01,
                "orient": tkinter.HORIZONTAL,
                "length": 200,
                "bg": "white",
                "highlightbackground": "white",
                "highlightthickness": 0,
                "troughcolor": "pink",
                "width": 10,
                "showvalue": 1 if self.show_values else 0,
            }
        else:
            return {
                "from_": 2.0,
                "to": 10.0,
                "resolution": 0.1,
                "orient": tkinter.HORIZONTAL,
                "length": 200,
                "bg": "white",
                "highlightbackground": "white",
                "highlightthickness": 0,
                "troughcolor": "pink",
                "width": 10,
                "showvalue": 1 if self.show_values else 0,
            }

    def get_azim_scale_kwargs(self) -> Dict[str, Any]:
        return {
            "from_": -180,
            "to": 180,
            "resolution": 0.1,
            "orient": tkinter.HORIZONTAL,
            "length": 200,
            "bg": "white",
            "highlightbackground": "white",
            "highlightthickness": 0,
            "troughcolor": "pink",
            "width": 10,
            "showvalue": 1 if self.show_values else 0,
        }

    def get_elev_scale_kwargs(self) -> Dict[str, Any]:
        return {
            "from_": -90,
            "to": 90,
            "resolution": 0.1,
            "orient": tkinter.HORIZONTAL,
            "length": 200,
            "bg": "white",
            "highlightbackground": "white",
            "highlightthickness": 0,
            "troughcolor": "pink",
            "width": 10,
            "showvalue": 1 if self.show_values else 0,
        }

    def get_smal_scale_kwargs(self) -> Dict[str, Any]:
        return {
            "from_": -5,
            "to": 5,
            "resolution": 0.01,
            "orient": tkinter.HORIZONTAL,
            "bg": "white",
            "troughcolor": "black",
            "width": 10,
            "length": 200,
            "borderwidth": 0,
            "showvalue": 1 if self.show_values else 0,
        }

    def get_parameters_scale_kwargs(self) -> Dict[str, Any]:
        return {
            "from_": -5,
            "to": 5,
            "resolution": 0.01,
            "orient": tkinter.HORIZONTAL,
            "bg": "white",
            "troughcolor": "pink",
            "width": 10,
            "length": 200,
            "borderwidth": 0,
            "showvalue": 1 if self.show_values else 0,
        }

    def get_stats_scale_kwargs(self) -> Dict[str, Any]:
        return {
            "from_": 0,
            "to": 50,
            "resolution": 0.1,
            "orient": tkinter.HORIZONTAL,
            "bg": "white",
            "troughcolor": "pink",
            "width": 10,
            "length": 200,
            "borderwidth": 0,
            "highlightbackground": "white",
            "highlightthickness": 0,
            "showvalue": 1 if self.show_values else 0,
        }

    @staticmethod
    def get_reset_button_kwargs() -> Dict[str, Any]:
        return {
            "activebackground": "black",
            "activeforeground": "white",
            "bg": "white",
            "fg": "black",
            "highlightbackground": "white",
        }


@hydra.main(config_path="../config", config_name="sliders_demo_py3d")
def main(cfg):
    app = SlidersApp(**cfg)
    app.create_application()


if __name__ == "__main__":
    main()
