import cv2
import json
import logging
import argparse
from pathlib import Path
from semantify.utils.general import get_min_max_values, get_logger, normalize_data


def main(args):
    """
    Visualize the data with the labels stats on the images

    Keyboard Options:
        q - quit
        d - delete the image and the json file
    """
    logger = get_logger(__name__)

    logger.info("calculating min max values")
    min_max_dict = get_min_max_values(args.working_dir)
    rglob_suffix = "*_front.png" if args.sides else "*.png"
    files_generator = sorted(list(Path(args.working_dir).rglob(rglob_suffix)), key=lambda x: int(x.stem.split("_")[0]))
    if args.from_img_idx is not None:
        suffix = f"{str(args.from_img_idx)}_front.png" if args.sides else f"{str(args.from_img_idx)}.png"
        idx = files_generator.index(Path(args.working_dir) / suffix)
        files_generator = files_generator[idx:]
    if args.to_img_idx is not None:
        suffix = f"{str(args.to_img_idx)}_front.png" if args.sides else f"{str(args.to_img_idx)}.png"
        idx = files_generator.index(Path(args.working_dir) / suffix)
        files_generator = files_generator[: idx + 1]

    logger.info("starting to iterate over images")
    for file_idx, file in enumerate(files_generator):

        logging.info(f"processing image {file.stem}")

        # read the image
        frontal_img = cv2.imread(file.as_posix())
        if args.sides:
            side_img = cv2.imread((file.parent / file.name.replace("front", "side")).as_posix())

        # load the json file
        image_suffix = "_front.png" if args.sides else ".png"
        labels_path = file.with_name(f"{file.name.replace(f'{image_suffix}', '_labels.json')}")
        with open(labels_path.as_posix(), "r") as f:
            labels = json.load(f)

        # write labels on the image vertically
        for idx, (key, value) in enumerate(labels.items()):

            # normalize the data
            value = normalize_data({key: value}, min_max_dict)[key]

            cv2.putText(
                frontal_img,
                f"{key}: {value[0][0] * 100:.2f}%",
                (10, 20 * idx + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        if args.sides:
            img = cv2.hconcat([frontal_img, side_img])
        else:
            img = frontal_img

        # show the image
        cv2.imshow(f"{file.stem} -> {file_idx}/{len(files_generator)}", img)
        key = cv2.waitKey(0)

        # press q to quit
        if key == ord("q"):
            break

        # press d to delete the image and the json files
        if key == ord("d"):
            logger.info(f"deleting {file.stem}")
            file_num = file.stem.split("_")[0]
            file.unlink()
            (file.parent / f"{file_num}.json").unlink()
            labels_path.unlink()
        cv2.destroyAllWindows()

    logger.info("done!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--working_dir",
        type=str,
        default="/home/nadav2/dev/data/CLIP2Shape/images/saved_by_me",
    )
    parser.add_argument("--from_img_idx", type=int, default=None)
    parser.add_argument("--to_img_idx", type=int, default=None)
    parser.add_argument("-s", "--sides", action="store_true", default=False, help="whether the data is multiview or not")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
