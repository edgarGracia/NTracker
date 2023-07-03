from collections import deque
from typing import Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig

from NTracker.utils.utils import compute_dist_matrix


class FeaturesTracker:
    """Tracker based on feature vectors.
    """

    def __init__(self, cfg: DictConfig):
        """Create the features tracker.

        Args:
            cfg (DictConfig): A configuration object.
        """
        self.num_instances = cfg.tracker.num_instances
        self.remove_bg = cfg.tracker.remove_background
        self.buffer_len = cfg.features_tracker.features_buffer
        self.features_thr = cfg.features_tracker.features_threshold
        self.features_distance = cfg.features_tracker.distance_function

        if (self.features_distance != "euclidean" and
                self.features_distance != "cosine"):
            raise NotImplementedError(self.features_distance)

        self.buffer: List[deque] = None
        self.current_features: Dict[int, np.ndarray] = {}

    def _init_buffer(self, features_len: int):
        """Initialize the features buffer.

        Args:
            features_len (tuple): Length of the features vectors.
        """
        self.buffer = [
            deque(
                [np.ones(features_len, dtype=np.float64)],
                maxlen=self.buffer_len
            )
            for _ in range(self.num_instances)
        ]

    def add_features(self, features: np.ndarray, key: int):
        """Add a new features vector.

        Args:
            features (np.ndarray): Features vector.
            key (int): Original instance key.
        """
        if self.buffer is None:
            self._init_buffer(len(features))
        self.current_features[key] = features

    def reset(self):
        """Clear the current features.
        """
        self.current_features = {}

    def compute_distance(
        self,
        key_list: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the distance matrix of the current and previous features.

        Args:
            key_list (List[int]): Sorted keys list of the added positions.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The normalized distance matrix and
            the threshold matrix indicating which assignations are impossible
        """
        fea_list = [self.current_features[k] for k in key_list]

        # Construct the previous features matrix
        fea_mat = np.empty(
            (
                self.buffer_len,
                len(fea_list),
                self.num_instances
            ), dtype=np.float64
        )
        for i in range(self.buffer_len):
            prev_fea = [
                self.buffer[j][min(i, len(self.buffer[j])-1)]
                for j in range(self.num_instances)
            ]
            fea_mat[i] = compute_dist_matrix(
                fea_list,
                prev_fea,
                metric=self.features_distance
            )

        dist_mat = fea_mat.sum(axis=0)
        dist_mat = dist_mat / self.buffer_len

        # Compute thresholds
        if self.features_thr is not None:
            thr_mat = dist_mat > self.features_thr
        else:
            thr_mat = np.full_like(dist_mat, False, dtype="bool")

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
        """Update the features buffer with the computed assignations.

       Args:
            key_list (List[int]): List of sorted keys. The same used on
                ``compute_distance``.
            curr_idx (List[int]): Current index list
                (see the linear-sum-assignment algorithm).
            prev_idx (List[int]): Previous index list
                (see the linear-sum-assignment algorithm).
        """
        fea_list = [self.current_features[k] for k in key_list]
        for ci, pi in zip(curr_idx, prev_idx):
            self.buffer[pi].append(fea_list[ci])
