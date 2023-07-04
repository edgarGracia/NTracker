import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from NTracker.tracking.run_tracker import run_tracker
from NTracker.utils.assignations_utils import load_assignations, save_assignations

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Run the tracker and tasks.

    Args:
        cfg (DictConfig): A configuration object.
    """
    if cfg.track:
        # Perform the tracking
        assignations = run_tracker(cfg)
        save_assignations(assignations)
    else:
        # Load assignations from file
        assignations = load_assignations(cfg.assignations_path)

    # Run tasks
    for task_params in cfg.tasks:
        logger.info(f"Running task: {task_params._target_}")
        task = instantiate(task_params, cfg=cfg, _recursive_=False)
        task.run()


if __name__ == "__main__":
    main()
