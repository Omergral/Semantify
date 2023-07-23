import os
from typing import Literal, Optional


def get_root_dir():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return root_dir


def append_to_root_dir(path: str) -> str:
    root_dir = get_root_dir()
    return os.path.join(root_dir, path)


def get_model_abs_path(
    model: Literal["smplx", "smpl", "flame", "smal"],
    specific: Optional[Literal["male", "female", "neutral", "expression", "shape"]] = None,
) -> str:
    root_dir = get_root_dir()
    if model in ["smplx", "smpl"]:
        assert specific in ["male", "female", "neutral"], "unrecognized model type"
        model_path = os.path.join(root_dir, "models_ckpts", model, f"{model}_{specific}.ckpt")
    elif model == "flame":
        assert specific in ["expression", "shape"], "unrecognized model type"
        model_path = os.path.join(
            root_dir,
            "models_ckpts",
            model,
            specific,
            f"{model}_{specific}.ckpt",
        )
    elif model == "smal":
        model_path = os.path.join(root_dir, "models_ckpts", model, f"{model}.ckpt")
    else:
        raise ValueError("unrecognized model type")
    return model_path
