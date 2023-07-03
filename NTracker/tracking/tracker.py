from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from scipy.optimize import linear_sum_assignment

from NTracker.tracking.features_tracker import FeaturesTracker
from NTracker.tracking.mask_iou_tracker import MaskIouTracker
from NTracker.tracking.position_tracker import PositionTracker
from NTracker.utils import utils


class Tracker:
    """Tracker of segmentation instances based on position, IoU and features,
    where the number of instances in the system is fixed.
    """

    def __init__(self, cfg: DictConfig):
        """Create the tracker.

        Args:
            cfg (DictConfig): A configuration object.
        """
        self.remove_bg = cfg.tracker.remove_background
        self.num_instances = cfg.tracker.num_instances

        self.pos_weight = cfg.tracker.pos_weight
        self.features_weight = cfg.tracker.features_weight
        self.iou_weight = cfg.tracker.iou_weight

        if self.pos_weight > 0:
            self.pos_tracker = PositionTracker(cfg)
        else:
            self.pos_tracker = None

        if self.features_weight > 0:
            self.features_tracker = FeaturesTracker(cfg)
        else:
            self.features_tracker = None
        
        if self.iou_weight > 0:
            self.mask_iou_tracker = MaskIouTracker(cfg)
        else:
            self.mask_iou_tracker = None

        self.current_keys = set()

        self.features_generator = instantiate(cfg.features_generator)

    def reset(self):
        """Clear the current tracking data
        """
        self.current_keys.clear()

        if self.pos_tracker is not None:
            self.pos_tracker.reset()

        if self.features_tracker is not None:
            self.features_tracker.reset()

        if self.mask_iou_tracker is not None:
            self.mask_iou_tracker.reset()

    def add_instance(
        self,
        mask: np.ndarray,
        bounding_box: Tuple[int, int, int, int],
        key: int,
        image: Optional[np.ndarray] = None,
        image_path: Optional[Path] = None
    ):
        """Add a new instance to the tracker.

        Args:
            mask (np.ndarray): Segmentation binary mask of shape (H, W).
            bounding_box (Tuple[int, int, int, int]): Bounding box
                (xmin, ymin, xmax, ymax).
            key (int): Unique identifier for the instance.
            image (Optional[np.ndarray], optional): numpy BGR image of shape
                (H, W, 3) and "uint8" dtype. Defaults to None.
            image_path (Optional[Path], optional): Path to the image.
                Defaults to None.
        """
        # Features
        if self.features_tracker is not None:
            if image is not None:
                image = utils.crop_image_box(image, bounding_box)
                if self.remove_bg:
                    mask_crop = utils.crop_image_box(mask, bounding_box)
                    image = utils.cut_mask_image(image, mask_crop)
            feature = self.features_generator.predict(
                image=image,
                image_path=image_path,
                key=key
            )
            self.features_tracker.add_features(feature, key)

        # Position
        if self.pos_tracker is not None:
            pos = np.array(utils.box_center(bounding_box))
            self.pos_tracker.add_position(pos, key)

        # IoU
        if self.mask_iou_tracker is not None:
            self.mask_iou_tracker.add_mask(mask, bounding_box, key)

        # Store the current instance key
        if key in self.current_keys:
            raise KeyError(f"Key ({key}) already seen ({self.current_keys})")
        self.current_keys.add(key)

    def re_assign(self) -> Dict[int, int]:
        """Compute the re-assignation indexes.

        Returns:
            Dict[int, int]: Mapping between the un-tracked keys and the
                assigned tracked keys (previous_key: new_key).
        """
        # Get the current ids
        curr_keys = sorted(list(self.current_keys))

        # Create the distance and threshold matrix
        dist_mat = np.zeros(
            (len(curr_keys), self.num_instances),
            dtype=np.float64
        )
        thr_mat = np.full(
            (len(curr_keys), self.num_instances),
            False,
            dtype="bool"
        )

        # Aggregate the tracker distances and thresholds
        if self.pos_tracker is not None:
            dist, thr = self.pos_tracker.compute_distance(curr_keys)
            dist_mat += (dist * self.pos_weight)
            thr_mat = thr_mat + thr

        if self.features_tracker is not None:
            dist, thr = self.features_tracker.compute_distance(curr_keys)
            dist_mat += (dist * self.features_weight)
            thr_mat = thr_mat + thr

        if self.mask_iou_tracker is not None:
            dist, thr = self.mask_iou_tracker.compute_distance(curr_keys)
            dist_mat += (dist * self.iou_weight)
            thr_mat = thr_mat + thr

        # Apply the Hungarian (linear-sum-assignment) algorithm
        curr_idx, prev_idx = linear_sum_assignment(dist_mat)

        # Apply threshold for non-permitted assignations
        valid = []
        for i, (ci, pi) in enumerate(zip(curr_idx, prev_idx)):
            if not thr_mat[ci, pi]:
                valid.append(i)
        curr_idx = curr_idx[valid]
        prev_idx = prev_idx[valid]

        # Update the trackers
        if self.pos_tracker is not None:
            self.pos_tracker.update_previous(curr_keys, curr_idx, prev_idx)
        if self.features_tracker is not None:
            self.features_tracker.update_previous(curr_keys, curr_idx, prev_idx)
        if self.mask_iou_tracker is not None:
            self.mask_iou_tracker.update_previous(curr_keys, curr_idx, prev_idx)

        return {curr_keys[ci]: pi for ci,pi in zip(curr_idx, prev_idx)}
