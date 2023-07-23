import clip
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from typing import List
from pathlib import Path
from semantify.utils.general import get_descriptors


def parse_args():
    parser = argparse.ArgumentParser(description="Generate clip scores")
    parser.add_argument(
        "--model_type", type=str, choices=["smplx", "smpl", "flame", "smal"], default="smplx", required=False
    )
    parser.add_argument(
        "--specific",
        type=str,
        choices=["shape", "expression", "male", "female", "neutral"],
        required=False,
        default=None,
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multiview", action="store_true")
    parser.add_argument("--imgs_dir", type=str, required=True, help="path to the created images directory")
    parser.add_argument(
        "--descriptors", nargs="+", type=str, required=False, default=None, help="list of descriptors to use"
    )
    return parser.parse_args()


def generate_clip_scores(device: str, multiview: bool, imgs_dir: str, descriptors: List[List[str]], **kwargs):

    labels = [[label] for label in descriptors]

    model, preprocess = clip.load("ViT-B/32", device=device)

    if multiview:
        files_generator = sorted(list(Path(imgs_dir).rglob("*front.png")), key=lambda x: int(x.stem.split("_")[0]))
    else:
        files_generator = sorted(list(Path(imgs_dir).rglob("*.png")), key=lambda x: int(x.stem.split("_")[0]))
    dir_length = len(files_generator)
    encoded_labels = {label[0]: clip.tokenize(label).to(device) for label in labels}

    for file in tqdm(files_generator, desc="generating clip scores", total=dir_length):

        json_path = file.parent / f"{file.stem.split('_')[0]}_labels.json"
        json_data = {}

        if json_path.exists():
            with open(json_path, "r") as f:
                json_data = json.load(f)

        encoded_frontal_image = preprocess(Image.open(file.as_posix())).unsqueeze(0).to(device)
        if multiview:
            try:
                encoded_side_image = (
                    preprocess(Image.open((file.parent / file.name.replace("front", "side")).as_posix()))
                    .unsqueeze(0)
                    .to(device)
                )
            except FileNotFoundError:
                print(f"Side image not found for {file.name}")
                continue

        with torch.no_grad():

            # get the mean value of the front and side images for each label
            for label, encoded_label in encoded_labels.items():
                if label in json_data:
                    continue
                front_score = model(encoded_frontal_image, encoded_label)[0].cpu().numpy()
                if multiview:
                    side_score = model(encoded_side_image, encoded_label)[0].cpu().numpy()
                    json_data[label] = ((front_score + side_score) / 2).tolist()
                else:
                    json_data[label] = front_score.tolist()

        with open(json_path, "w") as f:
            json.dump(json_data, f)


def main():
    args = parse_args()
    if args.descriptors is None:
        descriptors = get_descriptors(args.model_type, args.specific)
    else:
        descriptors = args.descriptors
    del args.descriptors
    generate_clip_scores(**vars(args), descriptors=descriptors)


if __name__ == "__main__":
    main()
