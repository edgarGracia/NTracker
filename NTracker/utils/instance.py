from copy import deepcopy
from typing import Optional, Tuple

import numpy as np


class Instance:

    def __init__(
        self,
        bounding_box: Optional[Tuple[int, int, int, int]] = None,
        score: Optional[float] = None,
        id: Optional[int] = None,
        mask: Optional[np.ndarray] = None,
        label: Optional[str] = None,
        label_id: Optional[int] = None,
        image_id: Optional[str] = None
    ):
        self.bounding_box = bounding_box
        self.score = score
        self.id = id
        self.mask = mask
        self.label = label
        self.label_id = label_id
        self.image_id = image_id

    def dict(self) -> dict:
        return deepcopy(
            {k: v for k, v in self.__dict__.items()
             if not k.startswith("_") and k != "dict"}
        )

    def __str__(self):
        return str(self.dict())
