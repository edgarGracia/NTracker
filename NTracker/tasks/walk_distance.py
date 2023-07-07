import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from omegaconf import DictConfig
from scipy.spatial.distance import euclidean

from NTracker.utils.path_utils import get_run_path

logger = logging.getLogger(__name__)


class WalkDistance:
    """Measure the walking distance of each instance.
    
    TODO: Measure real world distance
    """

    def __init__(
        self,
        cfg: DictConfig,
        output_file: Optional[Union[Path, str]] = "walk.json",
    ):
        """Create walk distance calculator object.

        Args:
            cfg (DictConfig): A configuration object.
            output_file (Optional[Union[Path, str]], optional): Output file
                path. Relative paths are appended to the run path. Defaults to
                "walk.json"
        """
        output_file = Path(output_file)

        self.cfg = cfg
        self.output_file = (output_file if output_file.is_absolute()
                            else get_run_path(output_file))

    def _measure_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Measure the distance between two points.

        Args:
            p1 (Tuple[int, int]): First 2D point.
            p2 (Tuple[int, int]): Second 2D point.

        Returns:
            float: The distance between the two points
        """
        return euclidean(p1, p2)

    def run(self, tracking_data: Dict[int, Dict[int, Dict[str, int]]]):
        """Run the instance visualizer task.

        Args:
            tracking_data (Dict[int, Dict[int, Dict[str, int]]]): Tracking data:
                ({tracked_id: {frame_n: {original_id: , x: ..., y: ...}}})
        """
        track_distances = {}
        for track_id, track_frames in tracking_data.items():
            frame_keys = sorted(list(track_frames.keys()))
            for i in range(1, len(frame_keys)):
                curr = track_frames[frame_keys[i]]
                prev = track_frames[frame_keys[i-1]]
                curr_pos = (curr["x"], curr["y"])
                prev_pos = (prev["x"], prev["y"])
                dist = self._measure_distance(curr_pos, prev_pos)
                track_distances[track_id] = track_distances.setdefault(
                    track_id, 0) + dist

        logger.info(f"Saving walk distances to {self.output_file}")
        self.output_file.write_text(json.dumps(track_distances))
