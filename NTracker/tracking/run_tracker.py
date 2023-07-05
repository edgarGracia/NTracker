import logging
from typing import Dict, Tuple

from omegaconf import DictConfig

from NTracker.tracking.tracker import Tracker
from NTracker.utils import tracking_utils
from NTracker.utils import structures

logger = logging.getLogger(__name__)


def run_tracker(cfg: DictConfig) -> Tuple(
    Dict[int, Dict[int, int]],
    Dict[int, Dict[int, Tuple[int, int]]]
):
    """Perform the tracking of instances.

    Args:
            cfg (DictConfig): A configuration object.

    Returns:
        Tuple(
            Dict[int, Dict[int, int]],
            Dict[int, Dict[int, Tuple[int, int]]]
        ): Assignations dict for each frame (original_key: tracked_key) and
            the tracked instances positions in each frame
            (tracked_key: {frame: (x, y)}).
    """
    logger.info(f"Tracking annotations on: {cfg.annotations_path}")

    num_instances = cfg.tracker.num_instances
    filter_n_instances = cfg.tracker.filter_n_instances
    init_instances = cfg.tracker.init_instances
    filter_score = cfg.tracker.filter_score

    # Tracker object
    tracker = Tracker(cfg)

    frame_assignations: Dict[int, Dict[int, int]] = {}
    instances_positions: Dict[int, Dict[int, Tuple[int, int]]] = {}
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

            # Save positions
            for orig_k, track_k in assignations.items():
                pos = structures.box_center(instances[orig_k].bounding_box)
                if track_k not in instances_positions:
                    instances_positions[track_k] = {img_i: pos}
                else:
                    instances_positions[track_k][img_i] = pos

    except KeyboardInterrupt:
        pass

    return frame_assignations, instances_positions
