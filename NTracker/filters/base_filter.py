from NTracker.utils.structures import Instance


class BaseFilter:

    def __init__(self):
        pass
    
    def filter(self, instance: Instance) -> bool:
        raise NotImplementedError
