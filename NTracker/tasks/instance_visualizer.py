import logging
from pathlib import Path
from typing import Dict, Optional, Union

from omegaconf import DictConfig

from NTracker.utils.image_utils import read_image, write_image
from NTracker.utils.path_utils import get_run_path
from NTracker.utils.tracking_utils import iterate_dataset
from NTracker.visualization import draw

logger = logging.getLogger(__name__)


class InstanceVisualizer:
    """Save an image with the tracked instances for each frame.
    """

    def __init__(
        self,
        cfg: DictConfig,
        output_path: Optional[Union[Path, str]] = "images",
        rename_output: bool = False,
        image_extension: Optional[str] = None,
        zero_fill: int = 10,
        overlay_path: Optional[Union[Path, str]] = None,
    ):
        """Create an instance visualizer object.

        Args:
            cfg (DictConfig): A configuration object.
            output_path (Optional[Union[Path, str]], optional): Output path
                where save the images. Relative paths are appended to the run
                path. Defaults to "images".
            rename_output (bool, optional): Rename the output images to a
                numerical counter. Defaults to False.
            image_extension (Optional[str], optional): Output image extension
                (e.g. ".jpg"). If None it will use the same extension as the
                input images. Defaults to None.
            zero_fill (int, optional): Number of zeros to prepend to the image
                file names. Defaults to 10.
            overlay_path (Optional[Union[Path, str]]): Path to an overlay image
                to show on top. Defaults to None.
        """
        output_path = Path(output_path)

        self.cfg = cfg
        self.output_path = (output_path if output_path.is_absolute()
                            else get_run_path(output_path))
        self.output_path.mkdir(exist_ok=True, parents=True)
        self.rename_output = rename_output
        self.image_extension = image_extension
        self.zero_fill = zero_fill
        self.overlay_path = overlay_path

    def run(self, tracking_data: Dict[int, Dict[int, Dict[str, int]]]):
        """Run the instance visualizer task.

        Args:
            tracking_data (Dict[int, Dict[int, Dict[str, int]]]): Tracking data:
                ({tracked_id: {frame_n: {original_id: , x: ..., y: ...}}})
        """
        logger.info(f"Saving images on: {self.output_path}")

        if self.overlay_path is not None:
            overlay = read_image(self.overlay_path)
        else:
            overlay = None

        for img_i, instances, img, img_path in iterate_dataset(self.cfg):
            out_img = draw.draw_tracking_frame(
                cfg=self.cfg,
                image=img,
                frame_i=img_i,
                tracking_data=tracking_data,
                instances=instances,
                overlay=overlay
            )

            ext = (img_path.suffix if self.image_extension is None
                   else self.image_extension)
            if self.rename_output:
                out_path = self.output_path / (
                    str(img_i).zfill(self.zero_fill) + ext)
            else:
                out_path = self.output_path / (img_path.stem + ext)
            assert out_path != img_path
            write_image(out_path, out_img)
