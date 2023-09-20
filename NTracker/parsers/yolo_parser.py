from pathlib import Path
from typing import List, Union

import cv2
import numpy as np

from NTracker.filters import filter_instance
from NTracker.filters.base_filter import BaseFilter
from NTracker.parsers.base_parser import BaseParser
from NTracker.utils import path_utils
from NTracker.utils.structures import Instance


class YoloParser(BaseParser):

    def __init__(
        self,
        base_path: Union[Path, str],
        image_width: int,
        image_height: int,
        has_score: bool = True,
        filters: List[BaseFilter] = [],
    ):
        """Create a parser for yolo txt files.

        Args:
            base_path (Union[Path, str]): Path to the annotations.
            image_width (int): The width of the input images.
            image_height (int): The height of the input images.
            has_score (bool, optional): Whether the annotations have the
                detection confidence. Defaults to True.
            filters (List[BaseFilter], optional): List of filters to apply to
                instances. Defaults to [].
        """
        super().__init__()
        self.base_path = Path(base_path)
        self.has_score = has_score
        self.image_width = image_width
        self.image_height = image_height
        self.filters = filters

    def list_annotations(self) -> List[Path]:
        """Get a sorted list of the annotations on the base path.

        Returns:
            List[Path]: A sorted list of the annotations files on the base path.
        """
        annotations_paths = [
            i for i in self.base_path.iterdir()
            if i.suffix.lower() == ".txt"
        ]
        return path_utils.sort_numerical_paths(annotations_paths)

    def read(self, file_path: Union[Path, str]) -> List[Instance]:
        """Read the annotations from one image.

        Args:
            file_path (Union[Path, str]): Path to the annotation file.

        Returns:
            List[Instance]: List of instances.
        """
        # YOLO format (normalized coordinates):
        #   class, x1, y1, x2, y2, ..., [conf]
        #   ...
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        instances = []
        for instance_i, line in enumerate(lines):
            polygon = [float(i) for i in line.split(" ")]
            cls = polygon.pop(0)
            if self.has_score:
                score = polygon.pop()
            else:
                score = 1
            polygon = np.array(polygon).reshape((-1,2))
            polygon[:,0] *= self.image_width
            polygon[:,1] *= self.image_height
            polygon = polygon.round().astype("int")
            polygon[:,0] = polygon[:,0].clip(0, self.image_width-1)
            polygon[:,1] = polygon[:,1].clip(0, self.image_height-1)
            xmin = int(polygon[:,0].min())
            ymin = int(polygon[:,1].min())
            xmax = int(polygon[:,0].max())
            ymax = int(polygon[:,1].max())
            mask = np.zeros((self.image_height, self.image_width), dtype="uint8")
            mask = cv2.fillPoly(mask, [polygon.reshape((-1, 1, 2))], 1) > 0
            
            instance = Instance(
                bounding_box=(xmin, ymin, xmax, ymax),
                score=score,
                id=instance_i,
                mask=mask,
                label_id=cls,
            )
            if filter_instance(self.filters, instance):
                instances.append(instance)

        return instances
