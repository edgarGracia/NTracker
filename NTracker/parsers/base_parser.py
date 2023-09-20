from pathlib import Path
from typing import List, Union

from NTracker.filters.base_filter import BaseFilter
from NTracker.utils.structures import Instance


class BaseParser:

    def __init__(self,
        base_path: Union[Path, str] = None,
        filters: List[BaseFilter] = []
    ):
        pass

    def list_annotations(self) -> List[Path]:
        raise NotImplementedError

    def read(self, file_path: Union[Path, str]) -> List[Instance]:
        raise NotImplementedError
