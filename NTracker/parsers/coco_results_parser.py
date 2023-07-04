import json
from pathlib import Path
from typing import List, Union

import pycocotools.mask as Mask

from NTracker.utils.structures import Instance
from NTracker.utils import path_utils


class CocoResultsParser:

    def __init__(self, base_path: Union[Path, str]):
        self.base_path = Path(base_path)

    def list_annotations(self) -> List[Path]:
        """Get a sorted list of the annotations on the base path.

        Returns:
            List[Path]: A sorted list of the annotations files on the base path.
        """
        annotations_paths = [
            i for i in self.base_path.iterdir()
            if i.suffix.lower() == ".json"
        ]
        return path_utils.sort_numerical_paths(annotations_paths)

    def read(self, file_path: Union[Path, str]) -> List[Instance]:
        """Read the annotations from one image.

        Args:
            file_path (Union[Path, str]): Path to the annotation file.

        Returns:
            List[Instance]: List of instances.
        """
        annotations = json.loads(Path(file_path).read_text())

        instances = []
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
