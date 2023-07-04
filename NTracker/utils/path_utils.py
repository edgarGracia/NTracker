import re
from pathlib import Path
from typing import List, Optional, Union

from hydra.core.hydra_config import HydraConfig


def extract_numeric_from_string(s: str) -> int:
    """Extract the numeric part of a string.

    Args:
        s (str): A string.

    Returns:
        int: The integer part of the string. e.g. "image_123_5.png" -> 1235
    """
    matches = re.findall(r"\d+", s)
    if matches:
        i = "".join(matches)
        return int(i)
    return 0


def sort_numerical_paths(paths: List[Path]) -> List[Path]:
    """Sort a list of path by the numerical order of the filenames.

    Args:
        paths (List[Path]): List of path.

    Returns:
        List[Path]: Numerical-ordered list of path.
    """
    return sorted(paths, key=lambda x: extract_numeric_from_string(x.name))


def get_run_path(sub_path: Optional[Union[Path, str]] = None) -> Path:
    """Get the current hydra run path.

    Args:
        sub_path (Optional[Union[Path, str]], optional): Optional sub path to
            append to the run path. Defaults to None.

    Returns:
        Path: The hydra run path.
    """
    run_path = Path(HydraConfig.get().runtime.output_dir)
    if sub_path:
        run_path = run_path.joinpath(sub_path)
    return run_path


def get_sibling_path(
    filename_or_path: Union[Path, str],
    dir_path: Union[Path, str],
    valid_extension: Optional[List[str]] = None
) -> List[Path]:
    """Retrieves the path of the files inside ``dir_path`` which have the same
    name stem as ``filename_or_path`` (filename without extension).

    Args:
        filename_or_path (Union[Path, str]): Original filename or path to a
            file. Its name will be used to get the paths on the ``dir_path``.
        dir_path (Union[Path, str]): Directory where search the sibling files.
        valid_extension (Optional[List[str]], optional) List of valid file
            extensions. Defaults to None.

    Returns:
        List[Path]: List of found files.
    """
    filename = Path(filename_or_path).stem + ".*"
    file_paths = list(dir_path.glob(filename))

    file_paths = [
        p for p in file_paths
        if p.suffix in valid_extension
    ] if valid_extension else file_paths
    
    return file_paths
