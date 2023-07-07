import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from NTracker.utils import path_utils
from NTracker.utils.image_utils import write_image
from NTracker.utils.path_utils import get_run_path
from NTracker.visualization import draw

logger = logging.getLogger(__name__)


def _process_image(data):
    (ann_path, img_path, tracking_data, img_i, annotations_parser, output_path,
     cfg, rename_output, image_extension, zero_fill) = data

    instances = annotations_parser.read(ann_path)
    instances = {i: x for i, x in enumerate(instances)}

    img = cv2.imread(str(img_path))
    assert img is not None, img_path

    if not cfg.visualization.img_background:
        img = np.full_like(img, cfg.visualization.img_bg_color)

    for tracked_id, frames_dict in tracking_data.items():
        if img_i not in frames_dict:
            continue
        img = draw.draw_instance(
            cfg=cfg,
            image=img,
            image_i=img_i,
            instance_key=tracked_id,
            instance=instances[frames_dict[img_i]["original_id"]],
            positions=frames_dict,
        )
    ext = img_path.suffix if image_extension is None else image_extension
    if rename_output:
        out_path = output_path / (str(img_i).zfill(zero_fill) + ext)
    else:
        out_path = output_path / (img_path.stem + ext)
    assert out_path != img_path
    write_image(out_path, img)


class InstanceVisualizerMultiProcess:
    """Save an image with the tracked instances for each frame using multiple
    processes.
    """

    def __init__(
        self,
        cfg: DictConfig,
        output_path: Optional[Union[Path, str]] = None,
        folder_name: Union[Path, str] = "images",
        processes: Optional[int] = None,
        rename_output: bool = False,
        image_extension: Optional[str] = None,
        zero_fill: int = 10
    ):
        """Create an instance visualizer object.

        Args:
            cfg (DictConfig): A configuration object.
            output_path (Optional[Union[Path, str]], optional): Output parent
                path. If None the run path will be used. Defaults to None.
            folder_name (Union[Path, str], optional): Folder where save the
                images. Defaults to "images".
            processes (Optional[int], optional): Number of processes. If None
                the number total number of logical processors will be used.
                Defaults to None.
            rename_output (bool, optional): Rename the output images to a
                numerical counter. Defaults to False.
            image_extension (Optional[str], optional): Output image extension
                (e.g. ".jpg"). If None it will use the same extension as the
                input images. Defaults to None.
            zero_fill (int, optional): Number of zeros to prepend to the image
                file names. Defaults to 10.
        """
        self.cfg = cfg
        self.processes = processes
        self.output_path = (
            get_run_path(folder_name) if output_path is None
            else Path(output_path).joinpath(folder_name))
        self.output_path.mkdir(exist_ok=True, parents=True)
        self.rename_output = rename_output
        self.image_extension = image_extension
        self.zero_fill = zero_fill

    def run(self, tracking_data: Dict[int, Dict[int, Dict[str, int]]]):
        """Run the instance visualizer task.

        Args:
            tracking_data (Dict[int, Dict[int, Dict[str, int]]]): Tracking data:
                ({tracked_id: {frame_n: {original_id: , x: ..., y: ...}}})
        """
        logger.info(f"Saving images on: {self.output_path}")

        annotations_parser = instantiate(self.cfg.annotations_parser)
        annotations_paths = annotations_parser.list_annotations()

        # Set start and end frames
        start_frame = (self.cfg.start_frame
                       if self.cfg.start_frame is not None else 0)
        end_frame = (self.cfg.end_frame
                     if self.cfg.end_frame is not None else len(annotations_paths))

        images_path = Path(self.cfg.images_path)
        images_extensions = self.cfg.images_extensions

        data = []
        for ann_i, ann_path in enumerate(annotations_paths[start_frame:end_frame]):
            image_path = path_utils.get_sibling_path(
                ann_path, images_path, images_extensions)
            if not image_path:
                raise FileNotFoundError(
                    f"No image found for {ann_path} in {images_path}"
                )
            if len(image_path) > 1:
                logger.warning(
                    f"More than one image found for the annotation {ann_path} "
                    f"({image_path})"
                )
            data.append(
                (
                    ann_path,
                    image_path[0],
                    tracking_data,
                    ann_i,
                    annotations_parser,
                    self.output_path,
                    self.cfg,
                    self.rename_output,
                    self.image_extension,
                    self.zero_fill
                )
            )
        with Pool(self.processes) as pool:
            results = pool.imap_unordered(_process_image, data)
            pbar = tqdm(total=len(data))
            for _ in results:
                pbar.update()
