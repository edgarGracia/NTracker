import logging
from typing import Dict

from omegaconf import DictConfig

from NTracker.tracking.tracker import Tracker
from NTracker.utils import tracking_utils

logger = logging.getLogger(__name__)


def run_tracker(cfg: DictConfig) -> Dict[int, Dict[int, int]]:
    """Perform the tracking of instances.

    Args:
            cfg (DictConfig): A configuration object.

    Returns:
        Dict[int, Dict[int, int]]: Assignations dict for each frame
            (original_key: tracked_key).
    """
    logger.info(f"Tracking annotations on: {cfg.annotations_path}")

    num_instances = cfg.tracker.num_instances
    filter_n_instances = cfg.tracker.filter_n_instances
    init_instances = cfg.tracker.init_instances
    filter_score = cfg.tracker.filter_score

    # Tracker object
    tracker = Tracker(cfg)

    frame_assignations = {}
    try:
        for img_i, instances, image, image_path in \
                tracking_utils.iterate_dataset(cfg, cfg.tracker.load_images):

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
                    image=image,
                    image_path=image_path
                )
            assignations = tracker.re_assign()

            frame_assignations[img_i] = assignations

    except KeyboardInterrupt:
        pass

    return frame_assignations
