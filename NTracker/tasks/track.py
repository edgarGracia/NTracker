import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from NTracker.tracking.tracker import Tracker
from NTracker.utils import utils

logger = logging.getLogger(__name__)


class Track:
    """Perform the tracking of instances.
    """

    def __init__(self, cfg: DictConfig):
        """Create a Track object.

        Args:
            cfg (DictConfig): A configuration object.
        """
        self.cfg = cfg

    def run(self):
        """Run the tracking.
        """
        logger.info(f"Tracking annotations on: {self.cfg.annotations_path}")
        logger.info(f"Output path: {HydraConfig.get().runtime.output_dir}")

        images_path = Path(self.cfg.images_path)
        num_instances = self.cfg.tracker.num_instances
        filter_n_instances = self.cfg.tracker.filter_n_instances
        init_instances = self.cfg.tracker.init_instances
        filter_score = self.cfg.tracker.filter_score
        load_images = self.cfg.tracker.load_images

        # Tracker object
        tracker = Tracker(self.cfg)

        # Annotations parser
        annotations_parser = instantiate(self.cfg.annotations_parser)

        # List input images
        images_paths = [i for i in images_path.iterdir()
                        if i.suffix.lower() in self.cfg.images_extensions
                        ]
        images_paths = utils.sort_numerical_paths(images_paths)

        # Set start and end frames
        start_frame = (self.cfg.start_frame
                       if self.cfg.start_frame is not None else 0)
        end_frame = (self.cfg.end_frame
                     if self.cfg.end_frame is not None else len(images_paths))

        try:
            for img_i, img_path in enumerate(
                tqdm(images_paths[start_frame:end_frame], unit="img")
            ):
                # Read image
                if load_images:
                    img = utils.read_image(img_path)
                else:
                    img = None

                # Read annotation
                instances = annotations_parser.read(img_path)
                instances = {i: x for i, x in enumerate(instances)}

                # Filter num instances
                if filter_n_instances and len(instances) != num_instances:
                    continue
                if init_instances is not None:
                    if init_instances != len(instances):
                        continue
                    else:
                        init_instances = None

                # Filter score; get the instances with the best score
                if filter_score:
                    if num_instances != len(instances):
                        i_sort = sorted(
                            instances.items(),
                            key=lambda x: x[1].score,
                            reverse=True
                        )
                        instances = {
                            i[0]: i[1] for i in i_sort[:num_instances]
                        }

                # Track
                tracker.reset()
                for k, ins in instances.items():
                    tracker.add_instance(
                        mask=ins.mask,
                        bounding_box=ins.bounding_box,
                        key=k,
                        image=img,
                        image_path=img_path
                    )
                assignations = tracker.re_assign()
                instances = utils.re_assign_dict(instances, assignations)

                # TODO: Call tasks

        except KeyboardInterrupt:
            pass

    # TODO:
    # task_results = {t: t.end() for t in tasks}
    # image_saver.close()
    # return task_results
