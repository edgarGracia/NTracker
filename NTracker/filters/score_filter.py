from NTracker.utils.structures import Instance
from NTracker.filters.base_filter import BaseFilter


class ScoreFilter(BaseFilter):

    def __init__(
        self,
        score_threshold: float
    ):
        super().__init__()
        self.score_threshold = score_threshold
    
    def filter(self, instance: Instance) -> bool:
        if instance.score < self.score_threshold:
            return False
        return True
