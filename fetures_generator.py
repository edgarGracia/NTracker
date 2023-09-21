import argparse
import pickle
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from NTracker.utils import image_utils, tracking_utils


def main(config_path: Path, output_path: Path):
    output_path.mkdir(exist_ok=True, parents=True)

    # Get the cfg
    initialize_config_dir(
        config_dir=str(config_path.absolute().parent),
        job_name="features_generator",
        version_base=None,
    )
    cfg = compose(config_name=config_path.stem, overrides=[])

    # Create the features generator
    features_generator = instantiate(cfg.features_generator)

    # Iterate instances
    for img_i, instances, image, image_path in tracking_utils.iterate_dataset(
        cfg, True
    ):
        img_features = {}
        for k, ins in instances.items():
            img_crop = image_utils.crop_image_box(image, ins.bounding_box)
            if cfg.tracker.remove_background:
                mask_crop = image_utils.crop_image_box(ins.mask, ins.bounding_box)
                img_crop = image_utils.cut_mask_image(image, mask_crop)
            features = features_generator.predict(
                image=img_crop, image_path=image_path, key=k
            )
            img_features[k] = features

        with open(output_path / (image_path.stem + ".pkl"), "wb") as f:
            pickle.dump(img_features, f)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate and save features from " "a set of images and annotation"
    )
    ap.add_argument(
        "-o",
        "--output",
        help="Output path where save the features",
        type=Path,
        required=True,
    )
    default_config = Path(__file__).parent / "NTracker/conf/coco_track.yaml"
    ap.add_argument(
        "-c",
        "--config",
        help=f"Path to a configuration YAML. Defaults to {default_config}",
        type=Path,
        default=default_config,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.output)
