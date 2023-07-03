import json
from pathlib import List, Path

import pycocotools.mask as Mask

from NTracker.utils.instance import Instance


class CocoResultsParser:

    def __init__(self, base_path: Path):
        self.base_path = base_path

    def read(self, image_path: Path) -> List[Instance]:
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
                    # label=,
                    label_id=annot.get("category_id"),
                    image_id=annot.get("image_id")
                )
            )

        return instances
