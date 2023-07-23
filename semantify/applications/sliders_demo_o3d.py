import json
import torch
import tkinter
import hydra
import numpy as np
from pathlib import Path
from typing import Dict, Union, Any
from semantify.utils._3dmm_utils import ThreeDMMUtils
from semantify.utils.general import get_model_to_eval
from semantify.utils.models_factory import ModelsFactory


class SlidersApp:
    def __init__(self, cfg):

        self.root = None
        self.device = cfg.device
        self.texture = cfg.texture

        self.num_coeffs = None
        if "num_coeffs" in cfg:
            self.num_coeffs = cfg.num_coeffs

        self.on_parameters = cfg.on_parameters

        assert cfg.model_type in ["smplx", "flame", "smal"], "Model type should be smplx, flame or smal"
        self.model_type = cfg.model_type

        self.outpath = None
        if cfg.out_dir is not None:
            if not Path(cfg.out_dir).exists():
                Path(cfg.out_dir).mkdir(parents=True)
            try:
                img_id = int(sorted(list(Path(cfg.out_dir).glob("*.png")), key=lambda x: int(x.stem))[-1].stem) + 1
            except IndexError:
                img_id = 0
            self.outpath = Path(cfg.out_dir) / f"{img_id}.png"

        self.params = []
        self._3dmm_utils = ThreeDMMUtils()
        self.models_factory = ModelsFactory(self.model_type)
        self.gender = cfg.gender
        self.body_pose = torch.eye(3).expand(1, 21, 3, 3)
        self.with_face = cfg.with_face
        self.model_kwargs = self.models_factory.get_default_params(cfg.with_face, num_coeffs=self.num_coeffs)
        verts, faces, vt, ft = self.models_factory.get_model(**self.model_kwargs)
        renderer_kwargs = self.get_renderer_kwargs(verts, faces, vt, ft)
        self.renderer = self.models_factory.get_renderer(**renderer_kwargs)

        self.default_zoom_value = 0.7
        if self.model_type == "smal":
            self.renderer.default_zoom_value = 0.1

        self.renderer.render_mesh()
        self.production_scales = []
        self.initialize_params()

        self.ignore_random_jaw = True if cfg.model_type == "flame" and cfg.with_face else False

        if cfg.model_path is not None:
            self.model, labels = get_model_to_eval(cfg.model_path)
            self.mean_values = {label[0]: 20 for label in labels}
            self.input_for_model = torch.tensor(list(self.mean_values.values()), dtype=torch.float32)[None]

    def get_renderer_kwargs(self, verts, faces, vt, ft) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        return {
            "verts": verts,
            "faces": faces,
            "vt": vt,
            "ft": ft,
            "texture": self.texture,
            "paint_vertex_colors": True if self.model_type == "smal" else False,
        }

    def initialize_params(self):
        if self.on_parameters:
            if self.model_type == "smplx":
                self.betas = self.model_kwargs["betas"]
                self.expression = self.model_kwargs["expression"]
                self.params.append(self.betas)
                self.params.append(self.expression)

            elif self.model_type == "flame":
                if self.with_face:
                    self.face_expression = self.model_kwargs["expression_params"][..., :10]
                    self.jaw_pose = self.model_kwargs["jaw_pose"]
                    self.face_expression_for_scale = torch.cat([self.face_expression, self.jaw_pose], dim=1)
                    self.params.append(self.face_expression_for_scale)
                else:
                    self.face_shape = self.model_kwargs["shape_params"][..., :10]
                    self.params.append(self.face_shape)

            else:
                self.beta = self.model_kwargs["beta"]
                self.params.append(self.beta)

    def update_betas(self, idx: int):
        def update_betas_values(value: float):
            if isinstance(value, str):
                value = float(value)
            self.betas[0, idx] = value

            verts, _, _, _ = self._3dmm_utils.get_smplx_model(
                betas=self.betas, body_pose=self.body_pose, expression=self.expression
            )
            self.renderer.render_mesh(verts=verts)

        return update_betas_values

    def update_face_shape(self, idx: int):
        def update_face_shape_values(value: float):
            if isinstance(value, str):
                value = float(value)
            self.face_shape[0, idx] = value

            verts, _, _, _ = self._3dmm_utils.get_flame_model(shape_params=self.face_shape)

            self.renderer.render_mesh(verts=verts)

        return update_face_shape_values

    def update_face_expression(self, idx: int):
        def update_face_expression_values(value: float):
            if isinstance(value, str):
                value = float(value)
            self.face_expression_for_scale[0, idx] = value

            verts, _, _, _ = self._3dmm_utils.get_flame_model(
                expression_params=self.face_expression_for_scale[..., :10],
                jaw_pose=self.face_expression_for_scale[..., 10:],
            )

            self.renderer.render_mesh(verts=verts)

        return update_face_expression_values

    def update_beta_shape(self, idx: int):
        def update_beta_shape_values(value: float):
            if isinstance(value, str):
                value = float(value)
            self.beta[0, idx] = value

            verts, _, _, _ = self._3dmm_utils.get_smal_model(beta=self.beta)

            self.renderer.render_mesh(verts=verts)

        return update_beta_shape_values

    def update_labels(self, idx: int):
        def update_labels_values(value: float):
            if isinstance(value, str):
                value = float(value)
            self.input_for_model[0, idx] = value
            # print(self.input_for_model)  # for debug
            with torch.no_grad():
                out = self.model(self.input_for_model.to(self.device))
                if self.model_type == "smplx":
                    betas = out.cpu()
                    expression = torch.zeros(1, 10)
                    body_pose = torch.eye(3).expand(1, 21, 3, 3)
                    verts, _, _, _ = self._3dmm_utils.get_smplx_model(
                        betas=betas, body_pose=body_pose, expression=expression, gender=self.gender
                    )
                elif self.model_type == "flame":
                    if self.with_face:
                        verts, _, _, _ = self._3dmm_utils.get_flame_model(
                            expression_params=out.cpu(), jaw_pose=torch.tensor([[0.08]])
                        )
                    else:
                        verts, _, _, _ = self._3dmm_utils.get_flame_model(shape_params=out.cpu())

                else:
                    verts, _, _, _ = self._3dmm_utils.get_smal_model(beta=out.cpu())

                self.renderer.render_mesh(verts=verts)

        return update_labels_values

    def add_texture(self):
        if self.texture is not None:
            self.renderer.render_mesh(texture=self.texture)

    def remove_texture(self):
        self.renderer.remove_texture()

    def update_camera_zoom(self, value: float):
        if isinstance(value, str):
            value = float(value)
        self.renderer.zoom(value)

    def update_camera_rotation_x(self, value: float):
        if isinstance(value, str):
            value = float(value)
        self.renderer.rotate(x=value)

    def update_camera_rotation_y(self, value: float):
        if isinstance(value, str):
            value = float(value)
        self.renderer.rotate(y=value)

    def update_camera_rotation_z(self, value: float):
        if isinstance(value, str):
            value = float(value)
        self.renderer.rotate(z=value)

    def update_camera_translation_right(self, value: float):
        if isinstance(value, str):
            value = float(value)
        self.renderer.translate(right=value)

    def update_camera_translation_up(self, value: float):
        if isinstance(value, str):
            value = float(value)
        self.renderer.translate(up=value)

    def reset_camera(self):
        self.renderer.reset_camera()

    def get_key_for_model(self) -> str:
        return self.models_factory.get_key_name_for_model(self.model_type)

    def save_png(self):
        self.renderer.visualizer.capture_screen_image(self.outpath.as_posix())
        key = self.get_key_for_model()
        concat_params = self._zeros_to_concat()
        params = {key: [self.params[0].tolist()[0] + concat_params]}
        with open(self.outpath.with_suffix(".json"), "w") as f:
            json.dump(params, f)
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
        self.renderer.save_mesh(obj_path)
        if self.outpath is not None:
            new_img_id = int(self.outpath.stem) + 1
            self.outpath = self.outpath.parent / f"{new_img_id}.png"

    def random_button(self):
        if self.on_parameters:
            random_params = self.models_factory.get_random_params(self.with_face)[self.get_key_for_model()][0, :10]
            scales_list = self.production_scales if not self.ignore_random_jaw else self.production_scales[:-1]
            for idx, scale in enumerate(scales_list):
                scale.set(random_params[idx].item())

    def update_expression(self, idx: int):
        def update_expression_values(value: float):
            if isinstance(value, str):
                value = float(value)
            self.expression[0, idx] = value

            verts, _, _, _ = self._3dmm_utils.get_smplx_model(
                betas=self.betas, body_pose=self.body_pose, expression=self.expression
            )
            self.renderer.render_mesh(verts=verts)

        return update_expression_values

    def reset_parameters(self):
        if self.on_parameters:
            for scale in self.production_scales:
                scale.set(0)
        else:
            for label in self.production_scales:
                label.set(20)

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
        self.root.title("Text 2 Mesh")
        if self.on_parameters and self.model_type == "smplx":
            self.root.geometry("205x1000")
            button_canvas_coords = (30, 350)
        else:
            self.root.geometry("205x700")
            button_canvas_coords = (30, 100)

        main_frame = tkinter.Frame(self.root, bg="white")
        parameters_frame = tkinter.Frame(self.root, highlightbackground="white", highlightthickness=0, bg="white")
        buttons_frame = tkinter.Frame(self.root, highlightbackground="white", highlightthickness=0, bg="white")
        main_frame.pack(fill=tkinter.BOTH, expand=True)
        # ------------------------------------------------------------

        # ------------------------ Buttons --------------------------
        buttons_canvas = tkinter.Canvas(main_frame, bg="white", highlightbackground="white")
        buttons_canvas.pack(fill=tkinter.BOTH, expand=True, side=tkinter.BOTTOM)
        buttons_canvas.create_window(button_canvas_coords, window=buttons_frame, anchor=tkinter.NW)
        reset_button_kwargs = self.get_reset_button_kwargs()

        # all reset button
        reset_all_button = tkinter.Button(
            buttons_frame, text="Reset All", command=lambda: self.reset_parameters(), **reset_button_kwargs
        )
        reset_all_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)

        # add texture button
        add_texture_button = tkinter.Button(
            buttons_frame, text="Add Texture", command=lambda: self.add_texture(), **reset_button_kwargs
        )
        add_texture_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)

        # remove texture button
        remove_texture_button = tkinter.Button(
            buttons_frame, text="Remove Texture", command=lambda: self.remove_texture(), **reset_button_kwargs
        )
        remove_texture_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)

        # save image & params button
        save_img_n_params_button = tkinter.Button(
            buttons_frame, text="save png", command=lambda: self.save_png(), **reset_button_kwargs
        )
        save_img_n_params_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)

        # save image & params button
        save_obj_button = tkinter.Button(
            buttons_frame, text="save obj", command=lambda: self.save_obj(), **reset_button_kwargs
        )
        save_obj_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)

        # generate random params button
        random_params_button = tkinter.Button(
            buttons_frame, text="random params", command=lambda: self.random_button(), **reset_button_kwargs
        )
        random_params_button.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)
        # ------------------------------------------------------------

        # ----------------------- Parameters -------------------------
        parameters_canvas = tkinter.Canvas(main_frame, bg="white", highlightbackground="white")
        parameters_canvas.pack(fill=tkinter.BOTH, expand=True, side=tkinter.TOP)
        parameters_canvas.create_window((0, 0), window=parameters_frame, anchor=tkinter.NW)
        # ------------------------------------------------------------

        # ------------------- Parameters Scale Bars ------------------
        if self.on_parameters:

            if self.model_type == "smplx":
                scale_kwargs = self.get_parameters_scale_kwargs()
                if self.with_face:
                    for expression in range(self.expression.shape[1]):
                        expression_scale = tkinter.Scale(
                            parameters_frame,
                            label=f"expression {expression}",
                            command=self.update_expression(expression),
                            **scale_kwargs,
                        )
                        self.production_scales.append(expression_scale)
                        expression_scale.set(0)
                        expression_scale.pack()
                else:
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
                    for label in range(self.face_expression_for_scale.shape[1]):
                        label_tag = (
                            f"expression {label}"
                            if label != self.face_expression_for_scale.shape[1] - 1
                            else "jaw pose"
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
                            label=f"shape param {label}",
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
                        label=f"beta param {label}",
                        command=self.update_beta_shape(label),
                        **scale_kwargs,
                    )
                    self.production_scales.append(beta_shape_scale)
                    beta_shape_scale.set(0)
                    beta_shape_scale.pack()

        else:
            scale_kwargs = self.get_stats_scale_kwargs()
            for idx, (label, value) in enumerate(self.mean_values.items()):
                label_scale = tkinter.Scale(
                    parameters_frame,
                    label=label,
                    command=self.update_labels(idx),
                    **scale_kwargs,
                )
                self.production_scales.append(label_scale)
                label_scale.set(value)
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
        zoom_in_scale.set(self.default_zoom_value)
        zoom_in_scale.pack()

        self.root.mainloop()

    @staticmethod
    def get_zoom_scale_kwargs() -> Dict[str, Any]:
        return {
            "from_": 0.45,
            "to": 1,
            "resolution": 0.01,
            "orient": tkinter.HORIZONTAL,
            "length": 200,
            "bg": "white",
            "highlightbackground": "white",
            "highlightthickness": 0,
            "troughcolor": "black",
            "width": 3,
        }

    @staticmethod
    def get_smal_scale_kwargs() -> Dict[str, Any]:
        return {
            "from_": -5,
            "to": 5,
            "resolution": 0.01,
            "orient": tkinter.HORIZONTAL,
            "bg": "white",
            "troughcolor": "black",
            "width": 3,
            "length": 200,
        }

    @staticmethod
    def get_parameters_scale_kwargs() -> Dict[str, Any]:
        return {
            "from_": -2,
            "to": 5,
            "resolution": 0.01,
            "orient": tkinter.HORIZONTAL,
            "bg": "white",
            "troughcolor": "black",
            "width": 3,
            "length": 200,
        }

    @staticmethod
    def get_stats_scale_kwargs() -> Dict[str, Any]:
        return {
            "from_": 0,
            "to": 50,
            "resolution": 1,
            "orient": tkinter.HORIZONTAL,
            "bg": "white",
            "troughcolor": "black",
            "width": 3,
            "length": 200,
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


@hydra.main(config_path="../config", config_name="sliders_demo_o3d")
def main(cfg):
    app = SlidersApp(cfg.demo_kwargs)
    app.create_application()


if __name__ == "__main__":
    main()
