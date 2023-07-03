from typing import List, Union

import numpy as np


class Dummy:
    """Place holder for a features generator object.
    """

    def __init__(self):
        """Create the dummy generator.
        """
        pass

    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Return a zero array ([0]])
        """
        return np.zeros((1))

    def predict_batch(
        self,
        images: Union[List[np.ndarray], np.ndarray],
        **kwargs
    ) -> np.ndarray:
        """Return an array of zeros.
        """
        return np.zeros((len(images), 1))
