import cv2
import json
import torch
import argparse
import trimesh
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
from typing import Literal, List
from semantify.utils._3dmm_utils import ThreeDMMUtils
from semantify.utils.models_factory import ModelsFactory
from semantify.utils.general import get_model_to_eval, get_plot_shape, get_renderer_kwargs, get_logger


def args_parser():
    parser = argparse.ArgumentParser(description="Evaluate performance")
    parser.add_argument("--mapper_path", type=str, help="path to mapper ckpt")
    parser.add_argument("--out_path", type=str, help="path to save results")
    parser.add_argument(
        "--method", 
        type=str, 
        choices=["diff_coords", "L2"],
        help="method to evaluate - diff_coords means differential coordinates, L2 means L2 norm")
    parser.add_argument("--gender", type=str, choices=["male", "female", "neutral"], default="neutral")
    parser.add_argument("--model_type", type=str, choices=["smplx", "smpl", "flame", "smal"])
    parser.add_argument(
        "--optimize_feature",
        type=str,
        choices=["betas", "beta", "shape_params", "expression_params"],
        help="feature to optimize",
    )
    parser.add_argument("--min_value", type=float, help="min slider value", default=15.0)
    parser.add_argument("--max_value", type=float, help="max slider value", default=30.0)
    parser.add_argument("--effect_threshold", type=float, help="effect threshold", default=0.5)
    parser.add_argument("--color_map", type=str, help="color map", default="YlOrRd")
    return parser.parse_args()


class EvaluatePerformance:
    def __init__(
        self,
        mapper_path: str,
        out_path: str,
        method: Literal["diff_coords", "L2"],
        gender: Literal["male", "female", "neutral"],
        model_type: Literal["smplx", "smpl", "flame", "smal"],
        optimize_feature: Literal["betas", "beta", "shape_params", "expression_params"],
        renderer_kwargs: DictConfig,
        min_value: float = 15.0,
        max_value: float = 30.0,
        effect_threshold: float = 0.5,
        color_map: str = "YlOrRd",
    ):
        self._3dmm_utils = ThreeDMMUtils()
        self.model_type = model_type
        self.gender = gender
        self.method = method
        self.models_factory = ModelsFactory(self.model_type)
        self.min_value = min_value
        self.max_value: float = max_value
        self.effect_threshold: float = effect_threshold
        self.color_map = plt.get_cmap(color_map)
        self.view_angles = range(0, 360, 45)
        self.num_rows, self.num_cols = self.get_collage_shape()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.labels = get_model_to_eval(mapper_path)
        self.default_input = (torch.ones(1, len(self.labels), dtype=torch.float32) * 20.0).to("cuda")
        self.optimize_feature = optimize_feature

        model_name = Path(mapper_path).stem
        self.out_path = Path(out_path) / model_name
        self.out_path.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(__name__)
        self._load_renderer(renderer_kwargs)
        self._get_total_possible_idxs()

    def _load_renderer(self, kwargs):
        self.renderer = self.models_factory.get_renderer(**kwargs)

    def _get_total_possible_idxs(self):
        if self.model_type in ["smpl", "smplx"]:
            self.verts, self.faces, self.vt, self.ft = self.models_factory.get_model(
                gender=self.gender, get_smpl=self.model_type == "smpl"
            )
            self.verts += self._3dmm_utils.smplx_offset_numpy
            total_possible_verts = self.verts.shape[0]
        elif self.model_type == "flame":
            self.verts, self.faces, self.vt, self.ft = self.models_factory.get_model()
            total_possible_verts = self.verts.shape[1]
        self.total_possible_idxs = torch.range(0, total_possible_verts)

    def get_collage_shape(self):
        num_rows, num_cols = get_plot_shape(len(self.view_angles))[0]
        if num_rows > num_cols:
            return num_cols, num_rows
        return num_rows, num_cols

    def get_verts_diff_coords(self, verts: np.ndarray, mesh: trimesh.Trimesh) -> List[torch.Tensor]:
        g = nx.from_edgelist(mesh.edges_unique)
        one_ring = [list(g[i].keys()) for i in range(len(mesh.vertices))]
        verts_diff_coords = np.zeros_like(verts)
        for i, v in enumerate(verts):
            verts_diff_coords[i] = v - verts[one_ring[i]].mean(axis=0)
        return verts_diff_coords

    @staticmethod
    def adjust_rendered_img(img: torch.Tensor):
        img = np.clip(img.cpu().numpy()[0, ..., :3] * 255, 0, 255).astype(np.uint8)
        return img

    def get_collage(self, images_list: List[np.ndarray]) -> np.ndarray:
        imgs_collage = [cv2.cvtColor(rend_img, cv2.COLOR_RGB2BGR) for rend_img in images_list]
        collage = np.concatenate(
            [
                np.concatenate(imgs_collage[i * self.num_cols : (i + 1) * self.num_cols], axis=1)
                for i in range(self.num_rows)
            ],
            axis=0,
        )
        return collage

    def get_verts_faces_by_model_type(self, verts, faces):
        if self.model_type in ["smpl", "smplx"]:
            return verts, faces
        return verts[0], faces

    def evaluate(self):
        effected_vertices = None
        shared_vertices = []
        for label_idx, descriptor in tqdm(enumerate(self.labels), total=len(self.labels), desc="Evaluating"):

            temp_input = self.default_input.clone()
            diffs = []
            for value in [self.min_value, self.max_value]:

                temp_input[0, label_idx] = value
                with torch.no_grad():
                    out = self.model(temp_input.to(self.device))

                model_kwargs = {
                    self.optimize_feature: out.cpu(),
                    "gender": self.gender,
                }
                if self.model_type == "smpl":
                    model_kwargs["get_smpl"] = True
                verts, faces, _, _ = self.models_factory.get_model(**model_kwargs)
                verts, faces = self.get_verts_faces_by_model_type(verts, faces)
                if self.method == "diff_coords":
                    diff = self.get_verts_diff_coords(
                        verts, trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                    )
                else:
                    diff = verts
                diffs.append(diff)

            diffs = np.linalg.norm(diffs[1] - diffs[0], axis=-1)
            diffs = diffs / diffs.max()

            vertex_colors = self.color_map(diffs)[:, :3]
            effective_indices = np.where(diffs > self.effect_threshold)[0]

            vertex_colors = torch.tensor(vertex_colors).float().to(self.device)
            if vertex_colors.dim() == 2:
                vertex_colors = vertex_colors.unsqueeze(0)

            rend_imgs = []
            for angle in self.view_angles:
                rend_img = self.renderer.render_mesh(
                    self.verts,
                    self.faces,
                    self.vt,
                    self.ft,
                    texture_color_values=vertex_colors,
                    rotate_mesh={"degrees": float(angle), "axis": "y"},
                )
                rend_img = self.adjust_rendered_img(rend_img)
                rend_img = cv2.cvtColor(rend_img, cv2.COLOR_RGB2BGR)
                rend_imgs.append(rend_img)
            collage = self.get_collage(rend_imgs)
            collage = cv2.cvtColor(collage, cv2.COLOR_RGB2BGR)

            if effected_vertices is None:
                effected_vertices = effective_indices
            else:
                effected_vertices = np.union1d(effected_vertices, effective_indices)

            shared_vertices.append(effective_indices)
            cv2.imwrite((self.out_path / f"{descriptor[0]}.png").as_posix(), collage)

        iou = len(effected_vertices) / len(self.total_possible_idxs)
        overlap = 1 - (effected_vertices.shape[0] / np.concatenate(shared_vertices).shape[0])
        self.logger.info(f"IOU: {iou} | Overlap: {overlap}")
        json_data = {
            "iou": iou,
            "labels": self.labels,
            "effect_threshold": self.effect_threshold,
            "overlap": overlap,
        }
        with open(self.out_path / "descriptors.json", "w") as f:
            json.dump(json_data, f)


def main():
    args = args_parser()
    renderer_kwargs = get_renderer_kwargs(
        model_type=args.model_type,
        **{"background_color": [255.0, 255.0, 255.0],
           "texture_optimization": True}
        )
    evaluate_performance = EvaluatePerformance(renderer_kwargs=renderer_kwargs, **vars(args))
    evaluate_performance.evaluate()


if __name__ == "__main__":
    main()
