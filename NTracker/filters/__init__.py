from typing import List

from NTracker.filters.base_filter import BaseFilter
from NTracker.filters.position_filter import PositionFilter
from NTracker.filters.score_filter import ScoreFilter

from NTracker.utils.structures import Instance

def filter_instance(filters: List[BaseFilter], instance: Instance) -> bool:
    for filter in filters:
        if not filter.filter(instance):
            return False
    return True