import logging
from pathlib import Path
from typing import Optional, Union

import ffmpeg
from omegaconf import DictConfig

from NTracker.utils.path_utils import get_run_path

logger = logging.getLogger(__name__)


class CreateVideoFfmpeg:
    """Create a video from images using ffmpeg.
    """

    def __init__(
        self,
        cfg: DictConfig,
        images_path: Union[Path, str] = "images",
        images_extension: Optional[str] = None,
        output_file: Optional[Union[Path, str]] = "tracking.mp4",
        fps: int = 25
    ):
        """Create an ffmpeg video creator object.

        Args:
            cfg (DictConfig): A configuration object.
            images_path (Union[Path, str], optional): Input images path.
                Relative paths are appended to the run path.
                Defaults to "images".
            images_extension (Optional[str], optional): The input images
                extension. If None the extension of the first file found will
                be used. Defaults to None.
            output_file (Optional[Union[Path, str]], optional): Output file
                path. Relative paths are appended to the run path. Defaults to
                "tracking.mp4"
            fps (int, optional): Frames per second of the video. Defaults to
                24.
        """
        images_path = Path(images_path)
        output_file = Path(output_file)

        self.cfg = cfg
        self.images_path = (images_path if images_path.is_absolute()
                            else get_run_path(images_path))
        self.output_file = (output_file if output_file.is_absolute()
                            else get_run_path(output_file))
        self.images_extension = images_extension
        self.fps = fps

    def run(self, *args, **kwargs):
        """Create the video.
        """
        ext = (self.images_extension if self.images_extension is not None
               else list(self.images_path.iterdir())[0].suffix)
        images_path = self.images_path / ("*" + ext)

        logger.info(f"Creating video {self.output_file} from {images_path}")
        (
            ffmpeg
            .input(images_path, pattern_type='glob', framerate=self.fps)
            .output(str(self.output_file))
            .run()
        )
