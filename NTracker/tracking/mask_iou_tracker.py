import concurrent
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig

from NTracker.utils.utils import box_intersect, mask_iou


class MaskIouTracker:
    """Tracker based on the intersection over union of the segmentation masks.
    """

    def __init__(self, cfg: DictConfig):
        """Create the IoU tracker.

        Args:
            cfg (DictConfig): A configuration object.
        """
        self.num_instances = cfg.tracker.num_instances

        self.prev_masks: np.ndarray = None
        self.prev_boxes = np.zeros((self.num_instances, 4), dtype="int")
        self.current_masks: Dict[int, np.ndarray] = {}
        self.current_boxes: Dict[int, np.ndarray] = {}

    def reset(self):
        """Clear the current data
        """
        self.current_masks = {}
        self.current_boxes = {}

    def _init_prev_masks(self, image_shape: tuple):
        """Initialize the previous masks array with the correct image size.

        Args:
            image_shape (tuple): Image size tuple
        """
        self.prev_masks = np.zeros(
            (self.num_instances, image_shape[0], image_shape[1]),
            dtype="bool"
        )

    def add_mask(
        self,
        mask: np.ndarray,
        bounding_box: np.ndarray,
        key: int
    ):
        """Add a new segmentation mask.

        Args:
            mask (np.ndarray): Segmentation mask numpy array of shape (H, W).
            bounding_box (np.ndarray): Bounding box of the mask
                (xmin, ymin, xmax, ymax).
            key (int): Original instance key.
        """
        if self.prev_masks is None:
            self._init_prev_masks(mask.shape)
        self.current_masks[key] = mask
        self.current_boxes[key] = bounding_box

    def compute_distance(
        self,
        key_list: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the distance matrix of the current and previous
        segmentation masks.

        Args:
            key_list (List[int]): Sorted keys list of the added positions.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The normalized distance matrix and
            the threshold matrix indicating which assignations are impossible.
        """
        masks_list = [self.current_masks[k] for k in key_list]
        boxes_list = [self.current_boxes[k] for k in key_list]

        # Compute the IoU between all the current and previous masks
        dist_mat = np.zeros(
            (len(masks_list), self.num_instances),
            dtype=np.float64
        )

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = {}
            for i in range(dist_mat.shape[0]):
                for j in range(dist_mat.shape[1]):
                    if not box_intersect(boxes_list[i], self.prev_boxes[j]):
                        continue
                    futures[
                        executor.submit(
                            mask_iou,
                            masks_list[i],
                            self.prev_boxes[j]
                        )
                    ] = (i, j)
        concurrent.futures.wait(futures)
        for f, (i, j) in futures.items():
            dist_mat[i, j] = f.result()

        # Dummy thresholds
        thr_mat = np.full_like(dist_mat, False, dtype="bool")

        # Inverse the values
        dist_mat = 1 - dist_mat

        return dist_mat, thr_mat

    def update_previous(
        self,
        key_list: List[int],
        curr_idx: List[int],
        prev_idx: List[int]
    ):
        """Update the previous segmentation masks with the computed assignations.

        Args:
            key_list (List[int]): List of sorted keys. The same used on
                ``compute_distance``.
            curr_idx (List[int]): Current index list
                (see the linear-sum-assignment algorithm).
            prev_idx (List[int]): Previous index list
                (see the linear-sum-assignment algorithm).
        """
        masks_list = [self.current_masks[k] for k in key_list]
        boxes_list = [self.current_boxes[k] for k in key_list]
        for ci, pi in zip(curr_idx, prev_idx):
            self.last_masks[pi] = masks_list[ci]
            self.last_boxes[pi] = boxes_list[ci]
