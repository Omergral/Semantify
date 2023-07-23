import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple, Literal, Optional
from semantify.utils._3dmm_utils import ThreeDMMUtils
from semantify.utils.image2shape import Image2ShapeUtils
from semantify.utils.paths_utils import get_model_abs_path
from semantify.utils.general import get_model_feature_name, get_model_to_eval, get_renderer_kwargs


def parse_args():
    parser = argparse.ArgumentParser(description="Image2Shape")
    parser.add_argument("--images_paths", type=str, nargs="+", help="Paths to images", required=True)
    parser.add_argument("--model_type", type=str, choices=["smplx", "smpl", "flame", "smal"], required=True)
    parser.add_argument("--specific", type=str, choices=["male", "female", "neutral", "expression", "shape"], default=None)
    parser.add_argument("--output_path", type=str, help="Output path", default="output/")
    args = parser.parse_args()
    return args


class Image2Shape(Image2ShapeUtils):
    def __init__(
        self,
        model_path: str,
        output_path: str,
        model_type: Literal["smplx", "smpl", "flame"],
        renderer_kwargs: Dict[str, Dict[str, float]],
        specific: Optional[Literal["male", "female", "neutral", "expression", "shape"]] = None,
        **kwargs,
    ):
        super().__init__()
        self.model_type = model_type
        self.specific = specific
        self.output_path: Path = Path(output_path)
        self._3dmm_utils: ThreeDMMUtils = ThreeDMMUtils(comparison_mode=True)
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        feature_name = get_model_feature_name(model_type, specific)
        model_path = {feature_name: model_path}

        self._load_renderer(renderer_kwargs)
        self.load_model(model_path)
        self._load_clip_model()
        self._encode_labels()

    def load_model(self, model_paths: Dict[str, str]):
        self.models = {}
        self.labels = {}
        for model_name, model_path in model_paths.items():
            model, labels = get_model_to_eval(model_path)
            self.models[model_name] = model
            self.labels[model_name] = self._flatten_list_of_lists(labels)

    def get_predictions(self, raw_img_path: Path) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:

        output_dict = {}

        # load raw image
        raw_img = cv2.imread(str(raw_img_path))
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        # clip preprocess
        encoded_image = self.clip_preprocess(Image.fromarray(raw_img)).unsqueeze(0).to(self.device)

        for model_name, model in self.models.items():
            # our prediction
            with torch.no_grad():
                clip_scores = self.clip_model(encoded_image, self.encoded_labels[model_name])[0].float()
                output_dict[model_name] = self.models[model_name](clip_scores).cpu()

        return output_dict, raw_img

    def get_rendered_images(
        self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], angle: float
    ) -> Dict[str, np.ndarray]:
        """Render the meshes for the different methods"""
        kwargs = {"rotate_mesh": {"degrees": float(angle), "axis": "y"}}
        rendered_img = self.renderer.render_mesh(*features, **kwargs)
        rendered_img = self.adjust_rendered_img(rendered_img)
        return rendered_img

    def __call__(self, image_path: Path):

        # get body shapes
        our_preds, raw_img = self.get_predictions(raw_img_path=Path(image_path))

        if self.model_type in ["smplx", "smpl"]:
            features = self._get_smplx_attributes(gender=self.specific, **our_preds)
        elif self.model_type in ["flame"]:
            features = self._3dmm_utils.get_flame_model(**our_preds)
        else:
            features = self._3dmm_utils.get_smal_model(**our_preds)

        # get rendered images
        rendered_img = self.get_rendered_images(features, angle=0)
        raw_img = cv2.resize(raw_img, rendered_img.shape[::-1][1:])

        # concatenate images
        concatenated_img = np.concatenate([raw_img, rendered_img], axis=1)
        concatenated_img = cv2.cvtColor(concatenated_img, cv2.COLOR_RGB2BGR)

        # save image
        out_path = self.output_path / f"pred_{Path(image_path).name}"
        cv2.imwrite(out_path.as_posix(), rendered_img)

        print()
        print("*" * 50)
        print("Image saved to: ", out_path.as_posix())
        print("*" * 50)


def main() -> None:
    args = parse_args()
    renderer_kwargs = get_renderer_kwargs(
        model_type=args.model_type,
        **{"background_color": [255.0, 255.0, 255.0]}
    )
    model_path = get_model_abs_path(args.model_type, args.specific)
    hbw_comparison = Image2Shape(
        model_path=model_path,
        model_type=args.model_type,
        renderer_kwargs=renderer_kwargs,
        specific=args.specific,
        output_path=args.output_path,
    )
    for image_path in args.images_paths:
        hbw_comparison(Path(image_path))


if __name__ == "__main__":
    main()
