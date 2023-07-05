import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from scipy.spatial import distance
from tqdm import tqdm

from NTracker.utils import image_utils, path_utils
from NTracker.utils.path_utils import get_run_path
from NTracker.utils.structures import Instance

logger = logging.getLogger(__name__)


def compute_dist_matrix(
    a: np.ndarray,
    b: np.ndarray,
    metric: Union[str, callable] = "euclidean"
) -> np.ndarray:
    """Compute the distance between two sets of vectors.

    Args:
        a (np.ndarray): Array of vectors of shape (m, c)
        b (np.ndarray): Array of vectors of shape (n, c)
        metric: (Union[str, callable], optional): The distance metric to use.
            See ``scipy.spatial.distance.cdist``.

    Returns:
        np.ndarray: A 2D matrix with the distance from each a-vector to each
        b-vectors, of shape (m, n)
    """
    return distance.cdist(a, b, metric=metric)


def iterate_dataset(
    cfg: DictConfig,
    load_images: bool = True,
    show_tqdm: bool = True
) -> Iterable[Tuple[int, Dict[int, Instance], Optional[np.ndarray], Optional[Path]]]:
    """Iterate through the annotations and images.

    Args:
        cfg (DictConfig): A configuration object.
        load_images (bool, optional): Load the images. Defaults to True.
        show_tqdm (bool, optional): Show a tqdm progress bar. Defaults to True.

    Yields:
        Iterable[Tuple[int, Dict[int, Instance], Optional[np.ndarray], Optional[Path]]]:
            Iterator that yields the frame number, a dict of Instances,
            the optional image and the image path.
    """
    # Annotations parser
    annotations_parser = instantiate(cfg.annotations_parser)
    annotations_paths = annotations_parser.list_annotations()

    # Set start and end frames
    start_frame = (cfg.start_frame
                   if cfg.start_frame is not None else 0)
    end_frame = (cfg.end_frame
                 if cfg.end_frame is not None else len(annotations_paths))

    # List input images
    if load_images:
        images_path = Path(cfg.images_path)
        images_extensions = cfg.images_extensions

    # Iterate through annotations and images
    for ann_i, ann_path in enumerate(tqdm(
        annotations_paths[start_frame:end_frame],
        unit="img",
        disable=not show_tqdm
    )):
        # Read image
        if load_images:
            image_path = path_utils.get_sibling_path(
                ann_path,
                images_path,
                images_extensions
            )
            if not image_path:
                raise FileNotFoundError(
                    f"No image found for {ann_path} in {images_path}"
                )
            if len(image_path) > 1:
                logger.warning(
                    f"More than one image found for the annotation {ann_path} "
                    f"({image_path})"
                )
            image = image_utils.read_image(image_path[0])
        else:
            image = None
            image_path = [None]

        # Read annotation
        instances = annotations_parser.read(ann_path)
        instances = {i: x for i, x in enumerate(instances)}
        yield ann_i+start_frame, instances, image, image_path[0]


def load_track(
    file_path: Union[Path, str]
) -> Dict[int, Dict[int, Dict[str, int]]]:
    """Load the tracking data from a file.

    Args:
        file_path (Union[Path, str]): Path to the tracking data JSON file.

    Returns:
        Dict[int, Dict[int, Dict[str, int]]]: Tracking data:
            {tracked_id: {frame_n: {original_id: ..., x: ..., y: ...}}}
    """
    logger.info(f"Loading tracking data from: {file_path}")
    return json.loads(Path(file_path).read_text())


def save_track(
    tracking_data: Dict[int, Dict[int, Dict[str, int]]],
    output_file: Optional[Union[Path, str]] = None
):
    """Save the tracking data to a JSON file.

    Args:
        tracking_data (Dict[int, Dict[int, Dict[str, int]]]): Tracking data:
            {tracked_id: {frame_n: {original_id: ..., x: ..., y: ...}}}
        output_file (Optional[Union[Path, str]], optional): Output file path.
            If None it will be set to a file in the run path. Defaults to None.
    """
    if output_file is None:
        output_file = get_run_path("track.json")
    logger.info(f"Saving tracking data to: {output_file}")
    Path(output_file).write_text(json.dumps(tracking_data))
