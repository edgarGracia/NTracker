import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union, List

from omegaconf import DictConfig
import cv2

from NTracker.utils.path_utils import get_run_path
from NTracker.utils.image_utils import read_image
from NTracker.utils.tracking_utils import iterate_dataset
from NTracker.utils.structures import mask_intersect

logger = logging.getLogger(__name__)



class TimeOnArea:
    """Measure the time an instance is touching a defined area.
    """

    def __init__(
        self,
        cfg: DictConfig,
        roi_paths: Union[List[Union[Path, str]], Union[Path, str]],
        output_path: Optional[Union[Path, str]] = "time_on_area",
        frame_time: float = 0,
        intersect_sources: Union[List[str], str] = "mask",
    ):
        """Create walk distance calculator object.

        Args:
            cfg (DictConfig): A configuration object.
            roi_paths (Union[List[Union[Path, str]], Union[Path, str]]):
                A single or a list of paths to a binary image delimiting the
                area of interest.
            output_path (Optional[Union[Path, str]], optional): Output folder.
                Relative paths are appended to the run path.
                Defaults to "time_on_area".
            frame_time (float, optional): Time passed between two
                consecutive frames in seconds. Defaults to 0.
            intersect_sources (Union[List[str], str], optional): Which source
                use to determine when an instance is intersecting the area of
                interest. One or a list. It must be one of
                ["mask", "box", "point"]. Defaults to "mask".
        """
        self.cfg = cfg
        self.frame_time = frame_time

        output_path = Path(output_path)
        self.output_path = (output_path if output_path.is_absolute()
                            else get_run_path(output_path))
        self.output_path.mkdir(parents=True, exist_ok=True)


        self.roi_names = []
        self.rois = []
        if not isinstance(roi_paths, (list, tuple)):
            roi_paths = [roi_paths]
        for path in roi_paths:
            roi = read_image(path)
            roi = roi.sum(axis=-1) > 0
            self.rois.append(roi)
            self.roi_names.append(path.stem)
        
        if isinstance(intersect_sources, str):
            self.intersect_sources = [intersect_sources] * len(self.rois)
        else:
            self.intersect_sources = intersect_sources

    def run(self, tracking_data: Dict[int, Dict[int, Dict[str, int]]]):
        """Run the instance visualizer task.

        Args:
            tracking_data (Dict[int, Dict[int, Dict[str, int]]]): Tracking data:
                ({tracked_id: {frame_n: {original_id: , x: ..., y: ...}}})
        """
        roi_time = [
            {ti: {"frames":0, "time":0} for ti in tracking_data.keys()}
            for _ in self.rois
        ]

        for img_i, instances, _, _ in iterate_dataset(self.cfg, False):
            for tracked_id, frames_dict in tracking_data.items():
                if img_i not in frames_dict:
                    continue
                instance = instances[frames_dict[img_i]["original_id"]]
                if self.intersect_source == "mask":
                    intersect = mask_intersect(self.roi, instance.mask)
                else:
                    raise NotImplementedError(self.intersect_source)
                if intersect:
                    roi_time[tracked_id]["frames"] += 1
                    roi_time[tracked_id]["time"] += self.frame_time
        
        logger.info(f"Saving time on area to {self.output_file}")
        self.output_file.write_text(json.dumps(roi_time))
