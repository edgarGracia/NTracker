from typing import Union, Optional, Dict
from pathlib import Path
from omegaconf import DictConfig

from NTracker.utils import Instance


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
        self.filename = filename
        self.output_path = output_path

    def run(self, frame_assignations: Dict[int, Dict[int, int]]):
        """Run the save positions task.

        Args:
            frame_assignations (Dict[int, Dict[int, int]]): Assignations dict
                for each frame (original_key: tracked_key).
        """
