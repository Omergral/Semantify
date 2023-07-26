from enum import Enum
from semantify.utils.paths_utils import append_to_root_dir


class TexturesPaths(Enum):
    SMPLX = append_to_root_dir("assets/smplx/smplx_texture.png")
    SMPL = append_to_root_dir("assets/smpl/smpl_texture.png")
    FLAME = append_to_root_dir("assets/flame/mean.npy")
    SMAL = None


class MaxCoeffs(Enum):
    """Enum for max coeffs for each model type."""

    SMPLX = 100
    SMPL = 100
    FLAME_SHAPE = 100
    FLAME_EXPRESSION = 50
    SMAL = 41
    JAW_POSE = 3


class VertsIdx(Enum):

    TOP_LIP_MIN = 3531
    TOP_LIP_MAX = 3532
    BOTTOM_LIP_MIN = 3504
    BOTTOM_LIP_MAX = 3505
