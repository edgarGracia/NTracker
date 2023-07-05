import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from omegaconf import DictConfig

from NTracker.utils.assignations_utils import re_assign_dict
from NTracker.utils.path_utils import get_run_path
from NTracker.utils.structures import box_center
from NTracker.utils.tracking_utils import iterate_dataset

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

    def run(
        self,
        frame_assignations: Dict[int, Dict[int, int]],
        instances_positions: Dict[int, Dict[int, Tuple[int, int]]]
        ):
        """Run the save positions task.

        Args:
            frame_assignations (Dict[int, Dict[int, int]]): Assignations dict
                for each frame (original_key: tracked_key).
            instances_positions (Dict[int, Dict[int, Tuple[int, int]]]):
                Instance position in each frame.
        """
        logger.info(f"Saving positions to: {self.output_path}")
        self.output_path.write_text(json.dumps(instances_positions))
