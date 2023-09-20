import json
from pathlib import Path
from typing import List, Union

import pycocotools.mask as Mask

from NTracker.filters import filter_instance
from NTracker.filters.base_filter import BaseFilter
from NTracker.parsers.base_parser import BaseParser
from NTracker.utils import path_utils
from NTracker.utils.structures import Instance


class CocoResultsParser(BaseParser):

    def __init__(
        self,
        base_path: Union[Path, str],
        filters: List[BaseFilter] = []
    ):
        """Create a parser for coco results json files.

        Args:
            base_path (Union[Path, str]): Path to the annotations.
            filters (List[BaseFilter], optional): List of filters to apply to
                instances. Defaults to [].
        """
        super().__init__()
        self.base_path = Path(base_path)
        self.filters = filters

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

            instance = Instance(
                bounding_box=box,
                score=annot.get("score"),
                id=annot.get("id"),
                mask=mask,
                label=None,
                label_id=annot.get("category_id"),
                image_id=annot.get("image_id")
            )

            if filter_instance(self.filters, instance):
                instances.append(instance)

        return instances
