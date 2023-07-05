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
        output_path: Optional[Union[Path, str]] = None
    ):
        """Create an instance visualizer object.

        Args:
            cfg (DictConfig): A configuration object.
            output_dir (Union[Path, str], optional): Output folder where
                save the images. Defaults to "images".
            output_path (Optional[Union[Path, str]], optional): Output parent
                path. If None the run path will be used. Defaults to None.
        """
        self.cfg = cfg
        self.output_path = (
            get_run_path(output_dir) if output_path is None
            else Path(output_path).joinpath(output_dir))
        self.output_path.mkdir(exist_ok=True, parents=True)

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
                    logger.warning(f"Frame {img_i} not tracked")
                    continue
                out_img = draw.draw_instance(
                    cfg=self.cfg,
                    image=img,
                    image_i=img_i,
                    instance_key=tracked_id,
                    instance=instances[frames_dict[img_i]["original_id"]],
                    positions=frames_dict,
                )
            out_path = self.output_path / img_path.name
            assert out_path != img_path
            write_image(out_path, out_img)
