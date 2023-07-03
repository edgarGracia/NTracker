from typing import Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig

from NTracker.utils.utils import compute_dist_matrix


class PositionTracker:
    """Tracker based on a single point position.
    """

    def __init__(self, cfg: DictConfig):
        """Create the position tracker.

        Args:
            cfg (DictConfig): A configuration object.
        """
        self.position_threshold = cfg.position_tracker.position_threshold
        self.square_position = cfg.position_tracker.square_position
        self.num_instances = cfg.tracker.num_instances

        self.prev_positions = np.full(
            (self.num_instances, 2),
            -1,
            dtype=np.int32
        )
        self.current_pos: Dict[int, List[int, int]] = {}

    def add_position(self, position: np.ndarray, key: int):
        """Add a new position.

        Args:
            pos (np.ndarray): Position coordinates ([x, y]).
            key (int): Original instance key.
        """
        self.current_pos[key] = position

    def reset(self):
        """Clear the current positions.
        """
        self.current_pos = {}

    def compute_distance(
        self,
        key_list: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the distance matrix of the current and previous positions.

        Args:
            key_list (List[int]): Sorted keys list of the added positions.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The normalized distance matrix and
            the threshold matrix indicating which assignations are impossible.
        """
        pos_list = [self.current_pos[k] for k in key_list]
        dist_mat = compute_dist_matrix(pos_list, self.prev_positions)

        # Compute thresholds
        if self.position_threshold is not None:
            thr_mat = dist_mat > self.position_threshold
            # Ignore the first-time cases
            thr_mat[:, np.all(self.prev_positions == -1, axis=1)] = False
        else:
            thr_mat = np.full_like(dist_mat, False, dtype="bool")

        if self.square_position:
            dist_mat = dist_mat**2

        # Normalize
        if np.any(dist_mat):
            dist_mat = dist_mat / dist_mat.max()

        return dist_mat, thr_mat

    def update_previous(
        self,
        key_list: List[int],
        curr_idx: List[int],
        prev_idx: List[int]
    ):
        """Update the previous positions with the computed assignations.

        Args:
            key_list (List[int]): List of sorted keys. The same used on
                ``compute_distance``.
            curr_idx (List[int]): Current index list
                (see the linear-sum-assignment algorithm).
            prev_idx (List[int]): Previous index list
                (see the linear-sum-assignment algorithm).
        """
        pos_list = [self.current_pos[k] for k in key_list]
        for ci, pi in zip(curr_idx, prev_idx):
            self.prev_positions[pi] = pos_list[ci]
