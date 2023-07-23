import torch
import smplx
import numpy as np
import pickle as pkl
from attrdict import AttrDict
from torch.nn import functional as F
from typing import Tuple, Optional, Union, Dict, Any, Literal
from scipy.spatial.transform import Rotation
from semantify.assets.smal.smal_layer import get_smal_layer
from semantify.assets.flame.flame import FLAME
from semantify.utils.enums import MaxCoeffs
from semantify.utils._3dmm_feats import SMALParams, SMPLXParams, FLAMEParams
from semantify.utils.paths_utils import append_to_root_dir


class ThreeDMMUtils:
    def __init__(self, device: str = "cuda", comparison_mode: bool = False):
        self.device = device
        self.body_pose = torch.tensor(np.load(append_to_root_dir("assets/smplx/a_pose.npy")))
        self.production_dir = append_to_root_dir("pre_production")
        self.comparison_mode = comparison_mode

    def _get_smplx_layer(self, gender: str, num_coeffs: int, get_smpl: bool):
        if get_smpl:
            if gender == "male":
                smplx_path = append_to_root_dir("assets/smpl/smpl_male.pkl")
            elif gender == "female":
                smplx_path = append_to_root_dir("assets/smpl/smpl_female.pkl")
            else:
                smplx_path = append_to_root_dir("assets/smpl/SMPL_NEUTRAL.pkl")
        else:
            if gender == "neutral":
                smplx_path = append_to_root_dir("assets/smplx/SMPLX_NEUTRAL_2020.npz")
            elif gender == "male":
                smplx_path = append_to_root_dir("assets/smplx/SMPLX_MALE.npz")
            else:
                smplx_path = append_to_root_dir("assets/smplx/SMPLX_FEMALE.npz")
        self.smplx_layer = smplx.build_layer(model_path=smplx_path, num_expression_coeffs=10, num_betas=num_coeffs)
        if get_smpl:
            self.smplx_faces = self.smplx_layer.faces_tensor
        else:
            model_data = np.load(smplx_path, allow_pickle=True)
            self.smplx_faces = model_data["f"].astype(np.int32)

    def _get_flame_layer(self, gender: Literal["male", "female", "neutral"]) -> FLAME:
        cfg = self.get_flame_model_kwargs(gender)
        self.flame_layer = FLAME(cfg).cuda()

    @property
    def smplx_offset_tensor(self):
        return torch.tensor([0.0, 0.4, 0.0], device=self.device)

    @property
    def smplx_offset_numpy(self):
        return np.array([0.0, 0.4, 0.0])

    @property
    def smpl_offset_numpy(self):
        return np.array([0.0, 0.4, 0.0])

    @property
    def smpl_offset_tensor(self):
        return torch.tensor([0.0, 0.4, 0.0], device=self.device)

    def get_smplx_model(
        self,
        betas: torch.Tensor = None,
        body_pose: torch.Tensor = None,
        expression: torch.Tensor = None,
        global_orient: torch.Tensor = None,
        transl: torch.Tensor = None,
        gender: Literal["neutral", "male", "female"] = "neutral",
        device: Optional[Literal["cuda", "cpu"]] = "cpu",
        num_coeffs: int = 10,
        get_smpl: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        smplx_model = SMPLXParams(
            betas=betas,
            body_pose=body_pose,
            expression=expression,
            global_orient=global_orient,
            transl=transl,
            smpl_model=get_smpl,
            num_coeffs=num_coeffs,
        )
        if self.comparison_mode:
            self._get_smplx_layer(gender, num_coeffs, get_smpl)
        else:
            if not hasattr(self, "smplx_layer") or not hasattr(self, "smplx_faces"):
                self._get_smplx_layer(gender, num_coeffs, get_smpl)

        if device == "cuda":
            smplx_model.params = smplx_model.to(device)
            self.smplx_layer = self.smplx_layer.cuda()
            verts = self.smplx_layer(**smplx_model.params).vertices
        else:
            verts = self.smplx_layer(**smplx_model.params).vertices
            verts = verts.detach().cpu().numpy()
        verts = verts.squeeze()
        if not hasattr(self, "vt_smplx") and not hasattr(self, "ft_smplx"):
            model_type = "smpl" if get_smpl else "smplx"
            self._get_vt_ft(model_type)

        return verts, self.smplx_faces, self.vt_smplx, self.ft_smplx

    def _get_vt_ft(self, model_type: Literal["smplx", "flame", "smpl"]) -> Tuple[np.ndarray, np.ndarray]:
        if model_type == "smplx":
            self.vt_smplx = np.load(append_to_root_dir("assets/smplx/textures/smplx_vt.npy"))
            self.ft_smplx = np.load(append_to_root_dir("assets/smplx/textures/smplx_ft.npy"))
        elif model_type == "smpl":
            self.vt_smplx = np.load(append_to_root_dir("assets/smplx/textures/smpl_uv_map.npy"))
            self.ft_smplx = self.smplx_faces
        else:
            flame_uv_path = append_to_root_dir("assets/flame/flame2020/flame_texture_data_v6.pkl")
            flame_uv = np.load(flame_uv_path, allow_pickle=True)
            self.vt_flame = flame_uv["vt_plus"]
            self.ft_flame = flame_uv["ft_plus"]

    def _get_flame_faces(self) -> np.ndarray:
        flame_uv_path = append_to_root_dir("assets/flame/flame2020/flame_texture_data_v6.pkl")
        flame_uv = np.load(flame_uv_path, allow_pickle=True)
        self.flame_faces = flame_uv["f_plus"]

    def _get_smal_faces(self) -> np.ndarray:
        smal_model_path = append_to_root_dir("assets/smal/smal_CVPR2017.pkl")
        with open(smal_model_path, "rb") as f:
            smal_model = pkl.load(f, encoding="latin1")
        self.smal_faces = smal_model["f"].astype(np.int32)

    @staticmethod
    def init_flame_params_dict(device: str = "cuda") -> Dict[str, torch.tensor]:
        flame_dict = {}
        flame_dict["shape_params"] = torch.zeros(1, 300)
        flame_dict["expression_params"] = torch.zeros(1, 100)
        flame_dict["global_rot"] = torch.zeros(1, 3)
        flame_dict["jaw_pose"] = torch.zeros(1, 3)
        flame_dict["neck_pose"] = torch.zeros(1, 3)
        flame_dict["transl"] = torch.zeros(1, 3)
        flame_dict["eye_pose"] = torch.zeros(1, 6)
        flame_dict["shape_offsets"] = torch.zeros(1, 5023, 3)
        flame_dict = {k: v.to(device) for k, v in flame_dict.items()}
        return flame_dict

    @staticmethod
    def get_flame_model_kwargs(gender: Literal["male", "female", "neutral"]) -> Dict[str, Any]:
        if gender == "male":
            flame_model_path = append_to_root_dir("assets/flame/flame2020/male_model.pkl")
        elif gender == "female":
            flame_model_path = append_to_root_dir("assets/flame/flame/female_model.pkl")
        else:
            flame_model_path = append_to_root_dir("assets/flame/flame2020/generic_model.pkl")

        kwargs = {
            "batch_size": 1,
            "use_face_contour": False,
            "use_3D_translation": True,
            "dtype": torch.float32,
            "device": torch.device("cpu"),
            "shape_params": 100,
            "expression_params": 50,
            "flame_model_path": flame_model_path,
            "ring_margin": 0.5,
            "ring_loss_weight": 1.0,
            "static_landmark_embedding_path": append_to_root_dir(
                "assets/flame/flame2020/flame_static_embedding_68.pkl"
            ),
            "pose_params": 6,
        }
        return AttrDict(kwargs)

    def get_flame_model(
        self,
        shape_params: torch.tensor = None,
        expression_params: torch.tensor = None,
        jaw_pose: float = None,
        gender: Literal["male", "female", "neutral"] = "neutral",
        device: Optional[Literal["cuda", "cpu"]] = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not hasattr(self, "flame_layer"):
            self._get_flame_layer(gender)
        if shape_params is not None and shape_params.shape == (1, 10):
            shape_params = torch.cat([shape_params, torch.zeros(1, 90).to(device)], dim=1)
        if expression_params is not None and expression_params.shape == (1, 10):
            expression_params = torch.cat([expression_params, torch.zeros(1, 40).to(device)], dim=1)
        flame_params = FLAMEParams(shape_params=shape_params, expression_params=expression_params, jaw_pose=jaw_pose)
        if device == "cuda":
            flame_params.params = flame_params.to(device)
        verts, _ = self.flame_layer(**flame_params.params)
        if device == "cpu":
            verts = verts.cpu()
        if not hasattr(self, "flame_faces"):
            self._get_flame_faces()

        if not hasattr(self, "vt_flame") and not hasattr(self, "ft_flame"):
            self._get_vt_ft("flame")

        return verts, self.flame_faces, self.vt_flame, self.ft_flame

    def get_smal_model(
        self, beta: torch.tensor, device: Optional[Literal["cuda", "cpu"]] = "cpu", py3d: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, None, None]:
        if not hasattr(self, "smal_layer"):
            self.smal_layer = get_smal_layer()
        smal_params = SMALParams(beta=beta)
        if device == "cuda":
            smal_params.params = smal_params.to(device)
            self.smal_layer = self.smal_layer.cuda()
            verts = self.smal_layer(**smal_params.params)[0]
        else:
            verts = self.smal_layer(**smal_params.params)[0].detach().cpu().numpy()
        verts = self.rotate_mesh_smal(verts, py3d)
        if not hasattr(self, "smal_faces"):
            self._get_smal_faces()
        return verts, self.smal_faces, None, None

    def get_body_pose(self) -> torch.Tensor:
        return self.body_pose

    @staticmethod
    def rotate_mesh_smal(verts: Union[np.ndarray, torch.Tensor], py3d: bool = True) -> np.ndarray:
        rotation_matrix_x = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        rotation_matrix_y = Rotation.from_euler("y", 75, degrees=True).as_matrix()
        rotation_matrix = np.matmul(rotation_matrix_x, rotation_matrix_y)
        if not py3d:
            rotation_matrix_z = Rotation.from_euler("x", -15, degrees=True).as_matrix()
            rotation_matrix = np.matmul(rotation_matrix, rotation_matrix_z)
        mesh_center = verts.mean(axis=1)
        if isinstance(verts, torch.Tensor):
            mesh_center = torch.tensor(mesh_center).to(verts.device).float()
            rotation_matrix = torch.tensor(rotation_matrix).to(verts.device).float()
        verts = verts - mesh_center
        verts = verts @ rotation_matrix
        verts = verts + mesh_center
        return verts

    @staticmethod
    def get_random_betas_smplx(num_coeffs: int = 10) -> torch.Tensor:
        """SMPLX body shape"""
        random_offset = torch.randint(-2, 2, (1, num_coeffs)).float()
        return torch.randn(1, num_coeffs) * random_offset

    @staticmethod
    def get_random_betas_smal(num_coeffs: int = 10) -> torch.Tensor:
        """SMAL body shape"""
        shape = torch.rand(1, num_coeffs) * torch.randint(-2, 2, (1, num_coeffs)).float()
        if num_coeffs < MaxCoeffs.SMAL.value:
            shape = torch.cat([shape, torch.zeros(1, MaxCoeffs.SMAL.value - num_coeffs)], 1)
        return shape

    @staticmethod
    def get_random_shape(num_coeffs: int = 10) -> torch.Tensor:
        """FLAME face shape"""
        shape = torch.randn(1, num_coeffs) * torch.randint(-2, 2, (1, num_coeffs)).float()
        if num_coeffs < MaxCoeffs.FLAME_SHAPE.value:
            shape = torch.cat([shape, torch.zeros(1, MaxCoeffs.FLAME_SHAPE.value - num_coeffs)], 1)
        return shape

    @staticmethod
    def get_random_expression_flame(num_coeffs: int = 10) -> torch.Tensor:
        """FLAME face expression"""
        expression = torch.randn(1, num_coeffs)  # * torch.randint(-3, 3, (1, num_coeffs)).float()
        if num_coeffs < MaxCoeffs.FLAME_EXPRESSION.value:
            expression = torch.cat([expression, torch.zeros(1, MaxCoeffs.FLAME_EXPRESSION.value - num_coeffs)], 1)
        return expression

    @staticmethod
    def get_random_jaw_pose() -> torch.Tensor:
        """FLAME jaw pose"""
        jaw_pose = F.relu(torch.randn(1, 1)) * torch.tensor([0.1])
        return jaw_pose

    @staticmethod
    def get_default_parameters(body_pose: bool = False, num_coeffs: int = 10) -> torch.Tensor:
        if body_pose:
            return torch.eye(3).expand(1, 21, 3, 3)
        return torch.zeros(1, num_coeffs)

    @staticmethod
    def get_default_face_shape() -> torch.Tensor:
        return torch.zeros(1, 100)

    @staticmethod
    def get_default_face_expression() -> torch.Tensor:
        return torch.zeros(1, 50)

    @staticmethod
    def get_default_shape_smal() -> torch.Tensor:
        return torch.zeros(1, 41)
