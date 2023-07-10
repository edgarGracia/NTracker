import logging
from typing import Dict, Tuple

from omegaconf import DictConfig

from NTracker.tracking.tracker import Tracker
from NTracker.utils import structures, tracking_utils

logger = logging.getLogger(__name__)


def run_tracker(cfg: DictConfig) -> Dict[int, Dict[int, Dict[str, int]]]:
    """Perform the tracking of instances.

    Args:
            cfg (DictConfig): A configuration object.

    Returns:
        Dict[int, Dict[int, Dict[str, int]]]: Tracking data:
            {tracked_id: {frame_n: {original_id: ..., x: ..., y: ...}}}
    """
    logger.info(f"Tracking annotations on: {cfg.annotations_path}")

    num_instances = cfg.tracker.num_instances
    filter_n_instances = cfg.tracker.filter_n_instances
    init_instances = cfg.tracker.init_instances
    filter_score = cfg.tracker.filter_score

    # Tracker object
    tracker = Tracker(cfg)

    tracking_data: Dict[int, Dict[int, Dict[str, int]]] = {}
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
            assignations = tracker.re_assign()  # (original_key: tracked_key)

            for orig_k, track_k in assignations.items():
                x, y = structures.box_center(instances[orig_k].bounding_box)
                tracking_data.setdefault(track_k, {})[img_i] = {
                    "original_id": orig_k,
                    "x": x,
                    "y": y
                }

    except KeyboardInterrupt:
        pass

    return tracking_data
