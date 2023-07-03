import logging
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as TVT
from torchreid.utils import FeatureExtractor

logger = logging.getLogger(__name__)


class TorchreidPredictor:
    """Features generator using the Torchreid library.
    """

    def __init__(
        self,
        model_name: str,
        model_path: Path,
        image_size: Tuple[int, int],
        norm_mean: Tuple[int, int, int] = [0, 0, 0],
        norm_std: Tuple[float, float, float] = [1., 1., 1.],
        device: str = "CPU"
    ):
        """Load the features generator model.

        Args:
            model_name (str): Name of the model.
            model_path (Path): Path to the model.
            image_size (Tuple[int, int]): Input image size.
            norm_mean (Tuple[int, int, int], optional): Image mean used to 
                normalize the images, in the range of (0, 1).
                Defaults to [0, 0, 0].
            norm_std (Tuple[float, float, float], optional): Image std used to
                normalize the images. Defaults to [1., 1., 1.].
            device (str, optional): Device where execute the model. "CPU" or
                "CUDA". Defaults to "CPU".
        """
        self.device = device.lower()

        logger.info(f"Loading model {model_path}")
        self.torchreid_extractor = FeatureExtractor(
            model_name=model_name,
            model_path=str(model_path),
            device=self.device
        )

        self.preprocess_transforms = TVT.Compose([
            TVT.Resize((image_size[0], image_size[1])),
            TVT.Normalize(mean=norm_mean, std=norm_std)
        ])

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Pre-process an image.

        Args:
            image (np.ndarray): numpy BGR image, of type "uint8" and shape
                (H, W, 3).

        Returns:
            np.ndarray: The pre-processed image.
        """
        x = torch.tensor(image).to(self.device).float()
        x = image.permute(2, 0, 1)
        x = self.preprocess_transforms(x)
        return x

    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Predict the features of an image.

        image (np.ndarray): numpy BGR image, of type "uint8" and shape
                (H, W, 3).

        Returns:
            np.ndarray: The output features.
        """
        image = self._preprocess(image)
        pred = self.torchreid_extractor.model(image)
        # TODO:
        print(pred.shape)
        return pred.detach().cpu().numpy()

    def predict_batch(
        self,
        images: Union[List[np.ndarray], np.ndarray],
        **kwargs
    ) -> np.ndarray:
        """Predict the features of a batch of images.

        Args:
            images (Union[List[np.ndarray], np.ndarray]): A list of numpy BGR
                images, of type "uint8" and shape (H, W, 3).

        Returns:
            np.ndarray: The output features for each image.
        """
        images = [self._preprocess(i) for i in images]
        images = torch.stack(images, dim=0)
        pred = self.torchreid_extractor.model(images)
        # TODO:
        print(pred.shape)
        return pred.detach().cpu().numpy()
