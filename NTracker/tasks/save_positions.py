from typing import Union, Optional, Dict, Tuple
from pathlib import Path
from omegaconf import DictConfig
import logging
import json


from NTracker.utils.structures import box_center
from NTracker.utils.path_utils import get_run_path
from NTracker.utils.tracking_utils import iterate_dataset
from NTracker.utils.assignations_utils import re_assign_dict
logger = logging.getLogger(__name__)


class SavePositions:
    """Save the position of the tracked instances on each frame.
    """

    def __init__(
        self,
        cfg: DictConfig,
        filename: str = "positions.json",
        output_path: Optional[Union[Path, str]] = None
    ):
        """Create a save positions object.

        Args:
            cfg (DictConfig): A configuration object.
            filename (str, optional): Name of the output file. Defaults to
                "positions.json"
            output_path (Optional[Union[Path, str]], optional): Output parent
                path. If None the run path will be used. Defaults to None.
        """
        self.cfg = cfg
        self.output_path = (
            get_run_path(filename) if output_path is None
            else Path(output_path).joinpath(filename))

    def run(self, frame_assignations: Dict[int, Dict[int, int]]):
        """Run the save positions task.

        Args:
            frame_assignations (Dict[int, Dict[int, int]]): Assignations dict
                for each frame (original_key: tracked_key).
        """
        positions: Dict[int, Dict[int, Tuple[int, int]]] = {}

        for img_i, instances, _, _ in iterate_dataset(self.cfg, False):
            if img_i not in frame_assignations:
                logger.warning(f"Frame {img_i} not tracked")
                continue
            instances = re_assign_dict(instances, frame_assignations[img_i])
            for ins_id, ins in instances.items():
                pos = box_center(ins.bounding_box)
                if ins_id not in positions:
                    positions[ins_id] = {img_i: pos}
            else:
                positions[ins_id][img_i] = pos

        self.output_path.write_text(json.dumps(positions))
