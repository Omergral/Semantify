import cv2
import json
import torch
import base64
import logging
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
from io import BytesIO
from torch import nn
from pathlib import Path
from hydra import compose, initialize
from typing import Tuple, Literal, List, Dict, Any, Optional, Union
from semantify.train.mapper_module import C2M
from semantify.utils.constants import DEFAULT_RENDERER_KWARGS, DESCRIPTORS


def video_to_frames(video_path: Union[str, Path]):
    video_path = Path(video_path)
    frames_dir = video_path.parent / f"{video_path.stem}_frames"
    frames_dir.mkdir(exist_ok=True)
    vidcap = cv2.VideoCapture(video_path.as_posix())
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(f"{frames_dir}/{count:05d}.png", image)
        success, image = vidcap.read()
        count += 1


def get_min_max_values(working_dir: str) -> Dict[str, Tuple[float, float, float]]:
    stats = {}
    min_max_dict = {}

    for file in Path(working_dir).rglob("*_labels.json"):
        with open(file.as_posix(), "r") as f:
            data = json.load(f)
        for key, value in data.items():
            if key not in stats:
                stats[key] = []
            stats[key].append(value)

    for key, value in stats.items():
        stats[key] = np.array(value)
        # show min and max
        min_max_dict[key] = (np.min(stats[key]), np.max(stats[key]), np.mean(stats[key]))
    return min_max_dict


def get_model_feature_name(
    model: Literal["smplx", "smpl", "flame", "smal"],
    specific: Optional[Literal["expression", "shape"]] = None,
) -> str:
    if model in ["smplx", "smpl"]:
        feature_name = "betas"
    elif model == "flame":
        assert specific in ["expression", "shape"], "unrecognized model type"
        if specific == "expression":
            feature_name = "expression_params"
        else:
            feature_name = "shape_params"
    elif model == "smal":
        feature_name = "beta"
    else:
        raise ValueError("unrecognized model type")
    return feature_name


def find_multipliers(value: int) -> list:
    """
    Description
    -----------
    finds all of the pairs that their product is the value
    Args
    ----
    value (int) = a number that you would like to get its multipliers
    Returns
    -------
    list of the pairs that their product is the value
    """
    factors = []
    for i in range(1, int(value**0.5) + 1):
        if value % i == 0:
            factors.append((i, value / i))
    return factors


def get_plot_shape(value: int) -> Tuple[Tuple[int, int], int]:
    """
    Description
    -----------
    given a number it finds the best pair of integers that their product
    equals the given number.
    for example, given an input 41 it will return 5 and 8
    """
    options_list = find_multipliers(value)
    if len(options_list) == 1:
        while len(options_list) == 1:
            value -= 1
            options_list = find_multipliers(value)

    chosen_multipliers = None
    min_distance = 100
    for option in options_list:
        if abs(option[0] - option[1]) < min_distance:
            chosen_multipliers = (option[0], option[1])

    # it is better that the height will be the largest value since the image is wide
    chosen_multipliers = (
        int(chosen_multipliers[np.argmax(chosen_multipliers)]),
        int(chosen_multipliers[1 - np.argmax(chosen_multipliers)]),
    )

    return chosen_multipliers, int(value)


def normalize_data(data: Dict[str, float], min_max_dict: Dict[str, Tuple[float, float, Any]]) -> Dict[str, float]:
    for key, value in data.items():
        min_val, max_val, _ = min_max_dict[key]
        data[key] = (value - min_val) / (max_val - min_val)
    return data


def flatten_list_of_lists(list_of_lists) -> List[Any]:
    return [l[0] for l in list_of_lists]


def flatten_dict_of_dicts(dict_of_dicts: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    flattened_dict = {}
    for value in dict_of_dicts.values():
        flattened_dict.update(value)
    return flattened_dict


def get_logger(name):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s: - %(message)s")
    logger = logging.getLogger(name=name)
    return logger


def create_metadata(metadata: Dict[str, torch.tensor], file_path: str):
    """
    Description
    -----------
    creates a json file with the metadata of the model

    Args
    ----
    metadata (Dict[str, torch.tensor]) = a dictionary with the metadata of the model
    file_path (str) = the path to save the metadata file
    """
    # write tensors to json
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            value = value.tolist()
        metadata[key] = value

    with open(file_path, "w") as f:
        json.dump(metadata, f)


def convert_str_list_to_float_tensor(strs_list: List[str]) -> torch.Tensor:
    stats = [float(stat) for stat in strs_list[0].split(" ")]
    return torch.tensor(stats, dtype=torch.float32)[None]


def filter_params_hack(ckpt: Dict[str, Any], convert_legacy: bool = False) -> Dict[str, Any]:
    """
    Description
    -----------
    filters the parameters of the model from the checkpoint

    Args
    ----
    ckpt (Dict[str, Any]) = the checkpoint of the model

    Returns
    -------
    hack (Dict[str, Any]) = the filtered parameters
    """
    hack = {key.split("model.")[-1]: ckpt["state_dict"][key] for key in ckpt["state_dict"].keys() if "model" in key}
    return hack


def get_model_to_eval(model_path: str, device: str = "cuda") -> Tuple[nn.Module, List[List[str]]]:
    """
    Description
    -----------
    loads a model and its metadata

    Args
    ----
    model_path (str) = the path to the model

    Returns
    -------
    model (nn.Module) = the model
    labels (List[List[str]]) = the labels of the model
    """
    model_meta_path = model_path.replace(".ckpt", "_metadata.json")
    with open(model_meta_path, "r") as f:
        model_meta = json.load(f)

    # kwargs from metadata
    labels = model_meta["labels"]
    model_meta.pop("labels")
    if "lr" in model_meta:
        model_meta.pop("lr")

    # load model
    ckpt = torch.load(model_path)
    convert_legacy = False if "fc_layers" in list(ckpt["state_dict"].keys())[0] else True
    filtered_params_hack = filter_params_hack(ckpt, convert_legacy=convert_legacy)
    model = C2M(**model_meta).to(device)
    model.load_state_dict(filtered_params_hack)
    model.eval()

    return model, labels


def plot_scatter_with_thumbnails(
    data_2d: np.ndarray,
    thumbnails: List[np.ndarray],
    labels: Optional[List[np.ndarray]] = None,
    title: str = "",
    figsize: Tuple[int, int] = (1200, 1200),
    mark_size: int = 40,
):
    """
    Plot an interactive scatter plot with the provided thumbnails as tooltips.
    Args:
    - data_2d: 2D array of shape (n_samples, 2) containing the 2D coordinates of the data points.
    - thumbnails: List of thumbnails to be displayed as tooltips, each thumbnail should be a numpy array.
    - labels: List of labels to be used for coloring the data points, if None, no coloring is applied.
    - title: Title of the plot.
    - figsize: Size of the plot.
    - mark_size: Size of the data points.
    Returns:
    - Altair chart object.
    """

    def _return_thumbnail(img_array, size=100):
        """Return a thumbnail of the image array."""
        image = Image.fromarray(img_array)
        image.thumbnail((size, size), Image.ANTIALIAS)
        return image

    def _image_formatter(img):
        """Return a base64 encoded image."""
        with BytesIO() as buffer:
            img.save(buffer, "png")
            data = base64.encodebytes(buffer.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{data}"

    dataframe = pd.DataFrame(
        {
            "x": data_2d[:, 0],
            "y": data_2d[:, 1],
            "image": [_image_formatter(_return_thumbnail(thumbnail)) for thumbnail in thumbnails],
            "label": labels,
        }
    )

    chart = (
        alt.Chart(dataframe, title=title)
        .mark_circle(size=mark_size)
        .encode(
            x="x", y=alt.Y("y", axis=None), tooltip=["image"], color="label"
        )  # Must be a list for the image to render
        .properties(width=figsize[0], height=figsize[1])
        .configure_axis(grid=False)
        .configure_legend(orient="top", titleFontSize=20, labelFontSize=10, labelLimit=0)
    )

    if labels is not None:
        chart = chart.encode(color="label:N")

    return chart.display()


def get_renderer_kwargs(model_type: Literal["flame", "smplx", "smpl", "smal"], **kwargs) -> Dict[str, Any]:
    """
    Description
    -----------
    gets the renderer kwargs from the config file

    Args
    ----
    model_type (Literal["flame", "smplx", "smpl", "smal"]) = the model type

    Returns
    -------
    renderer_kwargs (Dict[str, Any]) = the renderer kwargs
    """
    with initialize(config_path="../config/renderer_kwargs"):
        cfg = compose(config_name=model_type)
    for key, value in kwargs.items():
        if key in DEFAULT_RENDERER_KWARGS:
            DEFAULT_RENDERER_KWARGS[key] = value
    renderer_kwargs = {**cfg, **DEFAULT_RENDERER_KWARGS, "model_type": model_type}
    return renderer_kwargs


def get_sliders_limiters(
    model_type: Literal["flame", "smplx", "smpl", "smal"],
    specific: Optional[Literal["male", "female", "neutral", "expression", "shape"]] = None,
) -> Dict[str, List[float]]:
    """
    Description
    -----------
    gets the sliders limiters from the config file

    Args
    ----
    model_type (Literal["flame", "smplx", "smpl", "smal"]) = the model type
    specific (Optional[Literal["male", "female", "neutral", "expression", "shape"]]) = the specific type of the model

    Returns
    -------
    sliders_limiters (Dict[str, List[float]]) = the sliders limiters
    """
    with initialize(config_path="../config/sliders_limiters"):
        cfg = compose(config_name=model_type)
    sliders_limiters = cfg[specific] if specific is not None else cfg
    return sliders_limiters


def get_descriptors(
    model_type: Literal["flame", "smplx", "smpl", "smal"],
    specific: Optional[Literal["male", "female", "neutral", "expression", "shape"]] = None,
) -> List[str]:
    """
    Description
    -----------
    gets word descriptors from the config file

    Args
    ----
    model_type (Literal["flame", "smplx", "smpl", "smal"]) = the model type
    specific (Optional[Literal["male", "female", "neutral", "expression", "shape"]]) = the specific type of the model

    Returns
    -------
    List[str] = the descriptors
    """
    model_type = "smplx" if model_type == "smpl" else model_type
    return DESCRIPTORS[f"{model_type.upper()}_{specific.upper()}" if specific is not None else model_type.upper()]
