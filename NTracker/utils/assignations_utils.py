import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

from NTracker.utils.path_utils import get_run_path

logger = logging.getLogger(__name__)


def re_assign_dict(
    instances: Dict[int, any],
    assignations: Dict[int, int]
) -> Dict[int, any]:
    """Re-assign the keys of a dictionary based on an assignations dict.

    Args:
        instances (Dict[int, any]): Instances dict.
        assignations (Dict[int, int]): Assignations dict
            (original_key: tracked_key).

    Returns:
        Dict[int, any]: The instances dict but with the tracked keys.
    """
    return {n: instances[p] for p, n in assignations.items()}


def load_assignations(file_path: Union[Path, str]) -> Dict[int, int]:
    """Load assignations form a json file.

    Args:
        file_path (Union[Path, str]): Path to the assignations file.

    Returns:
        Dict[int, int]: Assignations dict (original_key: tracked_key).
    """
    logger.info(f"Loading assignations from: {file_path}")
    return json.loads(Path(file_path).read_text())


def save_assignations(
    assignations: Dict[int, int],
    output_file: Optional[Union[Path, str]] = None
):
    """Save an assignations dict to a json file.

    Args:
        assignations (Dict[int, int]): Assignations dict.
        output_file (Optional[Union[Path, str]]): Output file path. If None
            it will be set to a file in the run path.
    """
    if output_file is None:
        output_file = get_run_path("assignations.json")
    logger.info(f"Saving assignations to: {output_file}")
    Path(output_file).write_text(json.dumps(assignations))
