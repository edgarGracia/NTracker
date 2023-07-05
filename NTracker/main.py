import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from NTracker.tracking.run_tracker import run_tracker
from NTracker.utils.tracking_utils import save_track, load_track
from NTracker.utils.path_utils import get_run_path

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Run the tracker and tasks.

    Args:
        cfg (DictConfig): A configuration object.
    """
    logger.info(f"Run path: {get_run_path()}")
    if cfg.track:
        # Perform the tracking
        tracking_data = run_tracker(cfg)
        save_track(tracking_data)
    else:
        # Load assignations from file
        tracking_data = load_track(cfg.tracking_file)

    # Run tasks
    for task_params in cfg.tasks:
        logger.info(f"Running task: {task_params._target_}")
        task = instantiate(task_params, cfg=cfg, _recursive_=False)
        task.run(tracking_data)


if __name__ == "__main__":
    main()
