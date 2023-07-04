from typing import Dict, Union
from pathlib import Path
import json


def re_assign_dict(
    instances: Dict[int, any],
    assignations: Dict[int, int]
) -> Dict[int, any]:
    """Re-assign the keys of a dictionary based on an assignations dict.

    Args:
        instances (Dict[int, any]): Instances dict.
        assignations (Dict[int, int]): Assignations dict (previous_key: new_key).

    Returns:
        Dict[int, any]: The re-assigned ``instances`` dict.
    """
    return {n: instances[p] for p, n in assignations.items()}


def load_assignations(file_path: Union[Path, str]) -> Dict[int, int]:
    return json.loads(Path(file_path).read_text())


def save_assignations(output_file: Union[Path, str], assignations: Dict[int, int]):
    Path(output_file).write_text(json.dumps(assignations))
