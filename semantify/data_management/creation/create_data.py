import argparse
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from typing import Union, Literal, Optional
from semantify.utils.enums import VertsIdx
from semantify.utils.general import create_metadata, get_renderer_kwargs
from semantify.utils._3dmm_utils import ThreeDMMUtils
from semantify.utils.models_factory import ModelsFactory
from semantify.utils.renderers import Open3dRenderer, Pytorch3dRenderer


def args_parser():
    parser = argparse.ArgumentParser(description="create data")
    parser.add_argument("--output_path", type=str, required=True, help="path to save data")
    parser.add_argument("--use_tex", action="store_true", help="use texture - HIGHLY RECOMMENDED")
    parser.add_argument("--specific", type=str, choices=["male", "female", "neutral", "expression", "shape"], help="model's specific")
    parser.add_argument("--model_type", type=str, choices=["smpl", "smal", "smplx", "flame"], help="model type")
    parser.add_argument("--multiview", action="store_true", help="render multiview images")
    parser.add_argument("--img_tag", type=str, help="image tag (IMG_TAG.png)", default=None)
    parser.add_argument("--num_coeffs", type=int, help="number of coefficients", default=10)
    parser.add_argument("--num_of_imgs", type=int, help="number of images to create", default=100)
    parser.add_argument("--o3d", action="store_true", help="use open3d renderer instead of pytorch3d, not supported yet")
    parser.add_argument("--img_size", type=int, help="image size", default=1024)
    parser.add_argument("--background_color", type=float, nargs="+", help="background color", default=[0.0, 0.0, 0.0])
    return parser.parse_args()


class DataCreator:
    def __init__(
        self,
        output_path: str,
        model_type: Literal["smpl", "smal", "smplx", "flame"],
        specific: Literal["male", "female", "neutral", "expression", "shape"] = None,
        multiview: bool = False,
        img_tag: Optional[str] = None,
        num_coeffs: int = 10,
        num_of_imgs: int = 1000,
        renderer_type: Literal["pytorch3d", "open3d"] = "pytorch3d",
        renderer_kwargs: Optional[DictConfig] = None,
        **kwargs,
    ):
        # parameters from config
        self.multiview = multiview
        self.img_tag = img_tag
        self.with_face = specific == "expression" and model_type == "flame"
        self.num_coeffs = num_coeffs
        self.num_of_imgs = num_of_imgs
        self.output_path: Path = Path(output_path)
        self.model_type = model_type
        self.gender = specific if model_type in ["smpl", "smplx"] else "neutral"
        self.renderer_type: Literal["pytorch3d", "open3d"] = renderer_type
        self.get_smpl = True if self.model_type == "smpl" else False

        # utils
        self._3dmm_utils: ThreeDMMUtils = ThreeDMMUtils()
        self.models_factory: ModelsFactory = ModelsFactory(self.model_type)
        self._load_renderer(renderer_kwargs)

    def _load_renderer(self, kwargs):
        self.renderer: Union[Pytorch3dRenderer, Open3dRenderer] = self.models_factory.get_renderer(**kwargs)

    def __call__(self):
        # start creating data
        progress_bar = tqdm(range(self.num_of_imgs), total=self.num_of_imgs, desc="creating data")
        gen_img_num = 0
        while gen_img_num < self.num_of_imgs:

            # get image id
            try:
                img_id = (
                    int(
                        sorted(list(Path(self.output_path).glob("*.png")), key=lambda x: int(x.stem.split("_")[0]))[
                            -1
                        ].stem.split("_")[0]
                    )
                    + 1
                )
            except IndexError or ValueError:
                img_id = 0

            # set image name
            img_name = self.img_tag if self.img_tag is not None else str(img_id)

            # get random 3DMM parameters
            model_kwargs = self.models_factory.get_random_params(with_face=self.with_face, num_coeffs=self.num_coeffs)

            if self.get_smpl:
                model_kwargs["get_smpl"] = True

            # extract verts, faces, vt, ft
            verts, faces, vt, ft = self.models_factory.get_model(
                **model_kwargs, gender=self.gender, num_coeffs=self.num_coeffs
            )
            if self.model_type == "flame":
                # restrict mouth opening in order to avoid unrealistic meshes
                y_top_lip = verts[0, VertsIdx.TOP_LIP_MIN.value : VertsIdx.TOP_LIP_MAX.value, 1]
                y_bottom_lip = verts[0, VertsIdx.BOTTOM_LIP_MIN.value : VertsIdx.BOTTOM_LIP_MAX.value, 1]
                if y_top_lip - y_bottom_lip < 1e-3:
                    continue

            if self.model_type in ["smplx", "smpl"]:
                verts += self._3dmm_utils.smplx_offset_numpy

            # render mesh and save image
            if self.renderer_type == "open3d":
                self.renderer.render_mesh()
                self.renderer.visualizer.capture_screen_image(f"{self.output_path}/{img_name}.png")
                self.renderer.visualizer.destroy_window()
            else:
                if self.multiview:
                    for azim in [0.0, 90.0]:
                        img_suffix = "front" if azim == 0.0 else "side"
                        img = self.renderer.render_mesh(
                            verts=verts, faces=faces[None], vt=vt, ft=ft, rotate_mesh={"degrees": azim, "axis": "y"}
                        )
                        self.renderer.save_rendered_image(img, f"{self.output_path}/{img_name}_{img_suffix}.png")
                else:
                    img = self.renderer.render_mesh(verts=verts, faces=faces[None], vt=vt, ft=ft)
                    self.renderer.save_rendered_image(img, f"{self.output_path}/{img_name}.png")

            gen_img_num += 1
            progress_bar.update(1)

            create_metadata(metadata=model_kwargs, file_path=f"{self.output_path}/{img_name}.json")


def main():
    args = args_parser()
    renderer_kwargs = get_renderer_kwargs(py3d=not args.o3d, **vars(args))
    data_creator = DataCreator(renderer_kwargs=renderer_kwargs, **vars(args))
    data_creator()


if __name__ == "__main__":
    main()
