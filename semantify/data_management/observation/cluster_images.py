import cv2
import clip
import json
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Union, Tuple, List, Dict
from semantify.utils.general import get_logger, get_descriptors


def args_parser():
    parser = argparse.ArgumentParser(description="Cluster images")
    parser.add_argument("--images_dir", type=str, help="path to directory with images")
    parser.add_argument("--out_path", type=str, help="path to directory where to save json with clusters")
    parser.add_argument("--model_type", type=str, help="model type")
    parser.add_argument("--specific", type=str, help="specific model type")
    parser.add_argument("--plot_results", action="store_true", help="plot results")
    parser.add_argument("--topk_value", type=int, help="top k value", default=5)
    parser.add_argument("--kmax", type=int, help="max k value", default=10)
    return parser.parse_args()


class ClusterImages:
    """
    Clustering implementation was based on https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
    """
    def __init__(self, plot_results: bool = False, topk_value: int = 5, kmax: int = 10):
        self.max_images = 3500
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.plot_results = plot_results
        self.topk_value = topk_value
        self.kmax = kmax

        get_logger(__name__)

    def get_best_k(self, images_embedding: np.ndarray) -> int:
        wss_scores = self.calculate_WSS(images_embedding, self.kmax)
        sil_scores = self.calculate_silhouette_score(images_embedding, self.kmax)
        best_k = np.argmax(sil_scores) + 2
        if self.plot_results:
            self.plot_k_choosing(sil_scores, wss_scores, self.kmax)
        return best_k

    @staticmethod
    def plot_k_choosing(sil_scores: List[float], wss_scores: List[int], kmax: int) -> int:
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))

        ax.plot(range(1, kmax + 1), wss_scores, color="blue", marker="o", label="WSS")
        ax.set_xlabel("Number of clusters", fontsize=16)
        ax.set_yticks([])
        ax.set_xticks(range(1, kmax + 1))
        ax.set_ylim(0, np.max(wss_scores) + 1000)

        ax2 = ax.twinx()
        ax2.plot(range(2, kmax + 1), sil_scores, color="green", marker="x", label="Silhouette score")
        ax2.vlines(
            x=np.argmax(sil_scores) + 2,
            ymin=0.0,
            ymax=np.max(sil_scores),
            color="red",
            linestyles="dashed",
            label="Best k",
        )
        ax2.set_yticks([])

        fig.suptitle("Elbow Method vs Silhouette Method", fontsize=20)
        fig.legend(loc="upper right", ncol=3)
        fig.tight_layout()

        plt.show()

    @staticmethod
    def calculate_silhouette_score(points: np.ndarray, kmax: int = 10) -> List[float]:
        sil = []
        # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
        for k in range(2, kmax + 1):
            kmeans = KMeans(n_clusters=k).fit(points)
            labels = kmeans.labels_
            sil.append(silhouette_score(points, labels, metric="euclidean"))
        return sil

    @staticmethod
    def calculate_WSS(points: np.ndarray, kmax: int = 10) -> List[int]:
        sse = []
        for k in range(1, kmax + 1):
            kmeans = KMeans(n_clusters=k).fit(points)
            centroids = kmeans.cluster_centers_
            pred_clusters = kmeans.predict(points)
            curr_sse = 0

            # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
            for i in range(len(points)):
                curr_center = centroids[pred_clusters[i]]
                curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

            sse.append(curr_sse)
        return sse

    @staticmethod
    def write_words_clusters_to_json(possible_words: Dict[int, List[str]], json_path: Union[str, Path]):
        json_data = {str(i): possible_words[i] for i in possible_words.keys()}
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)

    def calc_top_descriptors(
        self, descriptors: List[str], preprocessed_images: np.ndarray, kmeans_labels: List[np.array]
    ) -> Dict[int, List[str]]:
        tokenized_labels = clip.tokenize(descriptors).to(self.device)
        labels_count_for_class = {
            class_id: {label: 0 for label in descriptors} for class_id in np.unique(kmeans_labels)
        }
        for class_id, image in tqdm(
            zip(kmeans_labels, preprocessed_images),
            desc="finding best descriptors for each cluster",
            total=preprocessed_images.__len__(),
        ):
            logits = self.model(image, tokenized_labels)[0]
            top_k_labels = [descriptors[i] for i in logits.topk(self.topk_value).indices[0]]
            for label in top_k_labels:
                labels_count_for_class[class_id][label] += 1 / kmeans_labels[kmeans_labels == class_id].shape[0]

        df = pd.DataFrame(labels_count_for_class)
        max_vals = df.idxmax(1)
        possible_words = {i: max_vals[max_vals == i].index.tolist() for i in np.unique(kmeans_labels)}

        return possible_words

    @staticmethod
    def cluster_images_w_kmeans(images_embedding: np.ndarray, n_clusters: int) -> List[np.array]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(images_embedding)
        return kmeans.labels_

    def get_encoded_images(self, images_dir: Union[Path, str]) -> Tuple[List[np.array], List[np.array], List[np.array]]:
        """
        Description:
        ------------
        This function encodes images using CLIP model and returns encoded images, preprocessed images and original images

        Args:
        -----
        images_dir: Union[Path, str]
            path to directory with images

        Returns:
        --------
        encoded_images: List[np.array]
            list of encoded images
        preprocessed_images: List[np.array]
            list of preprocessed images
        images: List[np.array]
            list of original images
        """
        images = []
        encoded_images = []
        preprocessed_images = []
        images_generator = [file for file in Path(images_dir).rglob("*.png") if "side" not in file.stem]

        for idx, image in enumerate(tqdm(images_generator, total=len(images_generator), desc="Encoding images")):
            images.append(cv2.imread(image.as_posix()))
            image = self.preprocess(Image.open(image)).unsqueeze(0).to(self.device)
            preprocessed_images.append(image)
            with torch.no_grad():
                image_features = self.model.encode_image(image).half().cpu().numpy()
            encoded_images.append(image_features)
            if idx == self.max_images:
                self.logger.warning(f"Reached max images: {self.max_images} - stopping encoding")
                break

        encoded_images = np.concatenate(encoded_images, axis=0)
        return encoded_images, preprocessed_images, images

    def cluster_images(
        self,
        images_dir: Union[Path, str],
        descriptors: List[str],
        out_path: Union[str, Path],
        json_name: str,
    ) -> None:
        if isinstance(images_dir, Path):
            images_dir = images_dir.as_posix()
        if isinstance(out_path, str):
            out_path = Path(out_path)
        encoded_images, preprocessed_images, _ = self.get_encoded_images(images_dir=images_dir)
        best_k = self.get_best_k(encoded_images)
        kmeans_labels = self.cluster_images_w_kmeans(encoded_images, n_clusters=best_k)

        # delete unnecessary variables to free GPU memory
        del encoded_images
        torch.cuda.empty_cache()

        possible_words = self.calc_top_descriptors(descriptors, preprocessed_images, kmeans_labels)
        self.write_words_clusters_to_json(possible_words, out_path / f"{json_name}.json")


def main():
    args = args_parser()
    clusterer = ClusterImages(plot_results=args.plot_results, topk_value=args.topk_value, kmax=args.kmax)
    descriptors = get_descriptors(args.model_type, args.specific)
    clusterer.cluster_images(
        images_dir=args.images_dir,
        descriptors=descriptors,
        out_path=args.out_path,
        json_name=args.images_dir.split("/")[-1],
    )


if __name__ == "__main__":
    main()
