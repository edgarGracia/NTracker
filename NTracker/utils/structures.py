from copy import deepcopy
from typing import Optional, Tuple

import numpy as np


class Instance:
    """Structure to hold the data of a detection instance to track.
    """

    def __init__(
        self,
        bounding_box: Optional[Tuple[int, int, int, int]] = None,
        score: Optional[float] = None,
        id: Optional[int] = None,
        mask: Optional[np.ndarray] = None,
        label: Optional[str] = None,
        label_id: Optional[int] = None,
        image_id: Optional[str] = None
    ):
        """Create and instance object.

        Args:
            bounding_box (Optional[Tuple[int, int, int, int]], optional):
                Bounding box delimiting the object within the image, in the
                format: (xmin, ymin, xmax, ymax). Defaults to None.
            score (Optional[float], optional): Detection score in the range
                [0, 1]. Defaults to None.
            id (Optional[int], optional): Original identifier of the instance.
                Defaults to None.
            mask (Optional[np.ndarray], optional): Numpy binary mask delimiting
                the object within the image, of type bool and shape (H, W).
                Defaults to None.
            label (Optional[str], optional): Detected label. Defaults to None.
            label_id (Optional[int], optional): Identifier of the label.
                Defaults to None.
            image_id (Optional[str], optional): Identifier of the image.
                Defaults to None.
        """
        self.bounding_box = bounding_box
        self.score = score
        self.id = id
        self.mask = mask
        self.label = label
        self.label_id = label_id
        self.image_id = image_id

    def dict(self) -> dict:
        """Return the instance's fields as a dict
        """
        return deepcopy(
            {k: v for k, v in self.__dict__.items()
             if not k.startswith("_") and k != "dict"}
        )

    def __str__(self):
        return str(self.dict())


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
