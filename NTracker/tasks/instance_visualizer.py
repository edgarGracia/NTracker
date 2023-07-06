import logging
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np

from omegaconf import DictConfig

from NTracker.utils.image_utils import write_image
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
        output_dir: Union[Path, str] = "images",
        output_path: Optional[Union[Path, str]] = None,
        rename_output: bool = False,
        image_extension: Optional[str] = None,
        zero_fill: int = 10
    ):
        """Create an instance visualizer object.

        Args:
            cfg (DictConfig): A configuration object.
            output_dir (Union[Path, str], optional): Output folder where
                save the images. Defaults to "images".
            output_path (Optional[Union[Path, str]], optional): Output parent
                path. If None the run path will be used. Defaults to None.
            rename_output (bool, optional): Rename the output images to a
                numerical counter. Defaults to False.
            image_extension (Optional[str], optional): Output image extension
                (e.g. ".jpg"). If None it will use the same extension as the
                input images. Defaults to None.
            zero_fill (int, optional): Number of zeros to prepend to the image
                file names. Defaults to 10.
        """
        self.cfg = cfg
        self.output_path = (
            get_run_path(output_dir) if output_path is None
            else Path(output_path).joinpath(output_dir))
        self.output_path.mkdir(exist_ok=True, parents=True)
        self.rename_output = rename_output
        self.image_extension = image_extension
        self.zero_fill = zero_fill

    def run(self, tracking_data: Dict[int, Dict[int, Dict[str, int]]]):
        """Run the instance visualizer task.

        Args:
            tracking_data (Dict[int, Dict[int, Dict[str, int]]]): Tracking data:
                ({tracked_id: {frame_n: {original_id: , x: ..., y: ...}}})
        """
        logger.info(f"Saving images on: {self.output_path}")

        for img_i, instances, img, img_path in iterate_dataset(self.cfg):
            if not self.cfg.visualization.img_background:
                img = np.full_like(img, self.cfg.visualization.img_bg_color)
            
            for tracked_id, frames_dict in tracking_data.items():
                if img_i not in frames_dict:
                    continue
                out_img = draw.draw_instance(
                    cfg=self.cfg,
                    image=img,
                    image_i=img_i,
                    instance_key=tracked_id,
                    instance=instances[frames_dict[img_i]["original_id"]],
                    positions=frames_dict,
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
