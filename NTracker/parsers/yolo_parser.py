import json
from pathlib import Path
from typing import List, Union

import numpy as np
import pycocotools.mask as Mask

from NTracker.utils import path_utils
from NTracker.utils.structures import Instance


class YoloParser:

    def __init__(self, base_path: Union[Path, str], image_width: int,
                 image_height: int, has_score: bool = True):
        """Create a YoloParser object.

        Args:
            base_path (Union[Path, str]): Path to the annotations path.
            image_width (int): The width of the input images.
            image_height (int): The height of the input images.
            has_score (bool, optional): Whether the annotations have the
                detection confidence. Defaults to True.
        """
        self.base_path = Path(base_path)
        self.has_score = has_score
        self.image_width = image_width
        self.image_height = image_height

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
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        instances = []
        for instance_i, line in enumerate(lines):
            polygon = [float(i) for i in line.split(" ")]
            cls = polygon.pop()
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
            mask = 

            instances.append(
                Instance(
                    bounding_box=(xmin, ymin, xmax, ymax),
                    score=score,
                    id=instance_i,
                    mask=mask,
                    label_id=cls,
                )
            )

        annotations = json.loads(Path(file_path).read_text())

        for annot in annotations:
            if "bbox" in annot:
                xmin, ymin, w, h = annot["bbox"]
                box = (round(xmin), round(ymin), round(xmin+w), round(ymin+h))
            else:
                box = None

            if "segmentation" in annot:
                mask = Mask.decode(annot["segmentation"]).astype("bool")
            else:
                mask = None

            instances.append(
                Instance(
                    bounding_box=box,
                    score=annot.get("score"),
                    id=annot.get("id"),
                    mask=mask,
                    label=None,
                    label_id=annot.get("category_id"),
                    image_id=annot.get("image_id")
                )
            )

        return instances
