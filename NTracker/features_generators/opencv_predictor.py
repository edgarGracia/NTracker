import logging
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OpencvPredictor:
    """Features generator from a CNN using Opencv.
    """

    def __init__(
        self,
        model_path: Path,
        image_size: Tuple[int, int],
        norm_mean: Tuple[float, float, float] = [0., 0., 0.],
        norm_std: Tuple[float, float, float] = [1., 1., 1.],
        channel_first: bool = True,
        mean_output: bool = True,
        flatten_output: bool = True,
        color_mode: str = "RGB",
        device: str = "CPU"
    ):
        """Load the features generator model.

        Args:
            model_path (Path): Path to a CNN model.
            image_size (Tuple[int, int]): Input image size.
            norm_mean (Tuple[float,float,float], optional): Image mean used to
                normalize the images, in the range of (0, 255).
                Defaults to [0,0,0].
            norm_std (Tuple[float,float,float], optional): Image std used to
                normalize the images. Defaults to [1.,1.,1.].
            channel_first (bool, optional): Whether the model expects the input
                images to be channel-first. Defaults to True.
            mean_output (bool, optional): Mean the output vector dimensions.
                Defaults to True.
            flatten_output (bool, optional): Flatten the output vector.
                Defaults to True.
            color_mode (str, optional): Image color mode. "RGB" or "BGR".
                Defaults to "RGB".
            device (str, optional): Device where execute the model. "CPU" or
                "CUDA". Defaults to "CPU".
        """
        self.image_size = ((image_size, image_size)
                           if isinstance(image_size, int) else image_size)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.mean_output = mean_output
        self.channel_first = channel_first
        self.flatten_output = flatten_output
        self.color_mode = color_mode
        self.device = device.lower()

        self.model: cv2.dnn.Net
        self._load_model(model_path)

    def _load_model(self, model_path: Path):
        """Load the model.

        Args:
            model_path (Path): Path to the model file.
        """
        logger.info(f"Loading model {model_path}")
        self.model = cv2.dnn.readNet(str(model_path))
        if "cuda" in self.device:
            try:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except Exception as e:
                logger.error(e, )

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Pre-process an image.

        Args:
            img (np.ndarray): numpy BGR image, of type "uint8" and shape
                (H, W, 3).

        Returns:
            np.ndarray: The pre-processed image.
        """
        x = cv2.resize(img, (self.image_size[0], self.image_size[1]))
        if self.color_mode == "RGB":
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x.astype(np.float32)
        x = (x - self.norm_mean) / self.norm_std
        if self.channel_first:
            x = np.moveaxis(x, -1, 0)
        return x

    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Predict the features of an image.

        Args:
            image (np.ndarray): numpy BGR image, of type "uint8" and shape
                (H, W, 3).

        Returns:
            np.ndarray: The output features.
        """
        image = self._preprocess(image)
        image = np.expand_dims(image, axis=0)
        self.model.setInput(image)
        pred = self.model.forward()[0]
        if self.mean_output:
            pred = np.mean(pred, axis=tuple(range(1, pred.ndim)))
        if self.flatten_output:
            pred = pred.flatten()
        return pred

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
        images = np.stack(images, axis=0)
        self.model.setInput(images)
        pred = self.model.forward()
        if self.mean_output:
            pred = np.mean(pred, axis=tuple(range(2, pred.ndim)))
        if self.flatten_output:
            pred = pred.reshape((pred.shape[0], -1))

        return pred
