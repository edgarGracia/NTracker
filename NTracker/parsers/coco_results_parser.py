import json
from pathlib import Path
from typing import List, Union

import pycocotools.mask as Mask

from NTracker.utils.instance import Instance


class CocoResultsParser:

    def __init__(self, base_path: Union[Path, str]):
        self.base_path = Path(base_path)

    def read(self, image_path: Union[Path, str]) -> List[Instance]:
        """Read the annotations from one image.

        Args:
            image_path (Union[Path, str]): Image path. Its name is used to get
                the correct annotation.

        Returns:
            List[Instance]: List of instances.
        """
        image_path = Path(image_path)
        annotations_path = self.base_path / (image_path.stem + ".json")
        annotations = json.loads(annotations_path.read_text())

        instances = []
        for annot in annotations:
            if "bbox" in annot:
                xmin, ymin, w, h = annot["bbox"]
                box = (xmin, ymin, xmin+w, ymin+h)
            else:
                box = None

            if "segmentation" in annot:
                mask = Mask.decode(annot["segmentation"])
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
