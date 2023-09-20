from pathlib import Path
from typing import Union, List
from collections.abc import Sequence

import numpy as np

from NTracker.utils.image_utils import read_image
from NTracker.utils.structures import Instance, box_center
from NTracker.filters.base_filter import BaseFilter


class PositionFilter(BaseFilter):

    def __init__(
        self,
        mask_path: Union[List[Union[Path, str]], Union[Path, str]]
    ):
        super().__init__()
        # Load masks
        masks = None
        if isinstance(mask_path, (str, Path)):
            mask_path = [mask_path]
        for path in mask_path:
            mask = read_image(path)
            if masks is None:
                masks = mask
            else:
                masks = np.append(masks, mask, axis=2)
        self.mask = masks.astype(np.int32).sum(2) > 0
    
    def filter(self, instance: Instance) -> bool:
        pos = box_center(instance.bounding_box)
        if self.mask[pos[1], pos[0]]:
            return False
        return True
