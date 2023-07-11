from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


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


def read_image(image_path: Union[Path, str]) -> np.ndarray:
    """Read an image from file.

    Args:
        image_path (Union[Path, str]): Path to the image file.

    Raises:
        IOError: If the image can not be read.

    Returns:
        np.ndarray: Numpy BGR image of shape (H, W, 3) and "uint8" type.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise IOError(f"Can not read image {str(image_path)}")
    return img


def write_image(path: Union[Path, str], image: np.ndarray):
    """Write an image to a file.

    Args:
        path (Union[Path, str]): Path to the image.
        image (np.ndarray): Numpy BGR image of shape (H, W, 3) and "uint8" type.
    """
    cv2.imwrite(str(path), image)


def resize_image(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> Tuple[np.ndarray, float, float]:
    """Resize an image to the given width and height.

    Args:
        image (np.ndarray): The image to resize.
        width (Optional[int], optional): Desired width.
            If None it will resize the image with the provided ``height``,
            maintaining the aspect ration. Defaults to None.
        height (Optional[int], optional): Desired height.
            If None it will resize the image with the provided ``width``,
            maintaining the aspect ration. Defaults to None.

    Returns:
        Tuple(np.ndarray, float, float): The resized image, the horizontal
            resize factor and the vertical resize factor.
    """
    if width is None and height is None:
        return image, 1, 1
    h, w, = image.shape[:2]
    if width is not None and height is not None:
        return cv2.resize(image, (width, height)), width / w, height / h
    if width is not None:
        f = width / w
    elif height is not None:
        f = height / h
    return (cv2.resize(image, None, fx=f, fy=f), f, f)
