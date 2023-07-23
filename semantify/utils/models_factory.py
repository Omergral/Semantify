import torch
import numpy as np
from typing import Tuple, Union, Dict, Literal
from semantify.utils._3dmm_utils import ThreeDMMUtils
from semantify.utils.renderers import Pytorch3dRenderer, Open3dRenderer


class ModelsFactory:
    def __init__(self, model_type: Literal["flame", "smplx", "smal", "smpl"]):
        self.model_type = model_type
        self._3dmm_utils = ThreeDMMUtils()

    def get_model(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.model_type == "smplx" or self.model_type == "smpl":
            return self._3dmm_utils.get_smplx_model(**kwargs)
        elif self.model_type == "flame":
            if "num_coeffs" in kwargs:
                kwargs.pop("num_coeffs")
            return self._3dmm_utils.get_flame_model(**kwargs)
        else:
            if "gender" in kwargs:
                kwargs.pop("gender")
            if "num_coeffs" in kwargs:
                kwargs.pop("num_coeffs")
            return self._3dmm_utils.get_smal_model(**kwargs)

    def get_default_params(self, with_face: bool = False, num_coeffs: int = 10) -> Dict[str, torch.tensor]:

        params = {}

        if self.model_type == "smplx" or self.model_type == "smpl":
            params["body_pose"] = self._3dmm_utils.get_default_parameters(body_pose=True)
            params["betas"] = self._3dmm_utils.get_default_parameters(num_coeffs=num_coeffs)
            expression = None
            if with_face:
                expression = self._3dmm_utils.get_default_face_expression()
            params["expression"] = expression

        elif self.model_type == "flame":
            params["shape_params"] = self._3dmm_utils.get_default_face_shape()
            params["expression_params"] = self._3dmm_utils.get_default_face_expression()

        else:
            params["beta"] = self._3dmm_utils.get_default_parameters()

        return params

    def get_vt_ft(self):
        return self._3dmm_utils.get_vt_ft(self.model_type)

    def get_renderer(self, py3d: bool = False, **kwargs) -> Union[Open3dRenderer, Pytorch3dRenderer]:
        if py3d:
            return Pytorch3dRenderer(**kwargs)
        return Open3dRenderer(**kwargs)

    def get_random_params(self, with_face: bool = False, num_coeffs: int = 10) -> Dict[str, torch.tensor]:
        params = {}
        if self.model_type in ["smplx", "smpl"]:
            params["betas"] = self._3dmm_utils.get_random_betas_smplx(num_coeffs)
        elif self.model_type == "flame":
            if with_face:
                params["expression_params"] = self._3dmm_utils.get_random_expression_flame(num_coeffs)
            else:
                params["shape_params"] = self._3dmm_utils.get_random_shape(num_coeffs)

        else:
            params["beta"] = self._3dmm_utils.get_random_betas_smal(num_coeffs)

        return params

    def get_key_name_for_model(self, with_face: bool = False) -> str:
        if self.model_type == "smplx":
            if with_face:
                return "expression"
            return "betas"
        elif self.model_type == "flame":
            if with_face:
                return "expression_params"
            return "shape_params"
        else:
            return "beta"
