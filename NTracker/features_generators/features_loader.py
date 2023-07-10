import logging
from pathlib import Path
from typing import List, Union, Optional
import pickle

import numpy as np

logger = logging.getLogger(__name__)


class FeaturesLoader:
    """Load features from files.
    """

    def __init__(
        self,
        base_path: Union[Path, str],
        mean_output: bool = True,
        flatten_output: bool = True
    ):
        """Create the features loader.

        Args:
            base_path (Path): Path to the folder with the features pkl files.
            mean_output (bool, optional): Mean the output vector dimensions.
                Defaults to True.
            flatten_output (bool, optional): Flatten the output vector.
                Defaults to True.
        """
        self.base_path = Path(base_path)
        self.mean_output = mean_output
        self.flatten_output = flatten_output

    def predict(self, image_path: Path, key: int, **kwargs) -> np.ndarray:
        """Load features from file.

        Args:
            image_path (Path): Image path.
            key (int): Instance key.

        Returns:
            np.ndarray: The loaded features.
        """
        with open(self.base_path / (image_path.stem + ".pkl"), "rb") as f:
            data = pickle.load(f)
        features = data[key]

        if self.mean_output:
            features = np.mean(features, axis=tuple(range(1, features.ndim)))
        if self.flatten_output:
            features = features.flatten()
        
        return features

    def predict_batch(
        self,
        image_path: Path,
        keys: Optional[List[int]] = None,
        **kwargs
    ) -> np.ndarray:
        """Load a set of features from a file.

        Args:
            image_path (Path): Path to the image.
            keys (Optional[List[int]]): List of keys to load.

        Returns:
            np.ndarray: The loaded features.
        """
        with open(self.base_path / (image_path.stem + ".pkl"), "rb") as f:
            data = pickle.load(f)
        
        if keys is None:
            features = np.array([data[k] for k in sorted(list(data.keys()))])
        else:
            features = np.array([data[k] for k in keys])

        if self.mean_output:
            features = np.mean(features, axis=tuple(range(2, features.ndim)))
        if self.flatten_output:
            features = features.reshape((features.shape[0], -1))

        return features
