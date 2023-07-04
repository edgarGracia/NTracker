import re
from pathlib import Path
from typing import List, Tuple, Union, Optional
from hydra.core.hydra_config import HydraConfig


import cv2
import numpy as np
from scipy.spatial import distance


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


def box_intersect(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int]
) -> bool:
    """Return True if two bounding boxes intersect. False otherwise.

    Args:
        box_a (Tuple[int, int, int, int]): Bounding box (xmin, ymin, xmax, ymax)
        box_b (Tuple[int, int, int, int]): Bounding box (xmin, ymin, xmax, ymax)

    Returns:
        bool: True if both boxes intersect.
    """
    a_xmin, a_ymin, a_xmax, a_ymax = box_a
    b_xmin, b_ymin, b_xmax, b_ymax = box_b

    if a_xmax < b_xmin or a_xmin > b_xmax:
        return False

    if a_ymax < b_ymin or a_ymin > b_ymax:
        return False

    return True


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute the intersection over union of two masks.

    Args:
        mask_a (np.ndarray): A segmentation mask of shape (h, w).
        mask_b (np.ndarray): A segmentation mask of shape (h, w).

    Returns:
        float: The intersection over union between the two masks.
    """
    a_area = np.count_nonzero(mask_a)
    b_area = np.count_nonzero(mask_b)

    if a_area == 0 or b_area == 0:
        return 0

    intersection = np.count_nonzero(mask_a & mask_b)

    iou = intersection / (a_area + b_area - intersection)
    return iou


def box_center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Returns the center position of a bounding box.

    Args:
        box (Tuple[int, int, int, int]): A bounding box (xmin, ymin, xmax, ymax).

    Returns:
        Tuple[int, int]: Center coordinates (x, y).
    """
    return (
        round((box[0] + box[2])/2),
        round((box[1] + box[3])/2)
    )


def crop_image_box(
    image: np.ndarray,
    bounding_box: Tuple[int, int, int, int]
) -> np.ndarray:
    """Crop an image with a bounding box.

    Args:
        image (np.ndarray): numpy image.
        bounding_box (Tuple[int, int, int, int]): Bounding box
            (xmin, ymin, xmax, ymax).

    Returns:
        np.ndarray: A crop from the image.
    """
    xmin, ymin, xmax, ymax = bounding_box
    xmin = int(np.clip(xmin, 0, image.shape[1]-1))
    ymin = int(np.clip(ymin, 0, image.shape[0]-1))
    xmax = int(np.clip(xmax, 0, image.shape[1]-1))
    ymax = int(np.clip(ymax, 0, image.shape[0]-1))
    return image[ymin:ymax, xmin:xmax]


def cut_mask_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove the background of an image with a segmentation mask.

    Args:
        img (np.ndarray): The image to segment.
        mask (np.ndarray): numpy mask of shape (H, W) and bool dtype.

    Returns:
        np.ndarray: The segmented image with a black background.
    """
    image = image.copy()
    image[mask == 0] = 0
    return image


def extract_numeric_from_string(s: str) -> int:
    """Extract the numeric part of a string.

    Args:
        s (str): A string.

    Returns:
        int: The integer part of the string. e.g. "image_123_5.png" -> 1235
    """
    matches = re.findall(r"\d+", s)
    if matches:
        i = "".join(matches)
        return int(i)
    return 0


def sort_numerical_paths(paths: List[Path]) -> List[Path]:
    """Sort a list of path by the numerical order of the filenames.

    Args:
        paths (List[Path]): List of path.

    Returns:
        List[Path]: Numerical-ordered list of path.
    """
    return sorted(paths, key=lambda x: extract_numeric_from_string(x.name))


def read_image(image_path: Union[Path, str]) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise IOError(f"Can not read image {str(image_path)}")
    return img


def get_run_path(sub_path: Optional[Union[Path, str]]) -> Path:
    run_path = Path(HydraConfig.get().runtime.output_dir)
    if sub_path:
        run_path = run_path.joinpath(sub_path)
    return run_path
