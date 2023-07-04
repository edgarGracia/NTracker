import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from NTracker.tasks import Track
from NTracker.utils.assignations import load_assignations, save_assignations
from NTracker.utils import utils

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Run the tracker and tasks.

    Args:
        cfg (DictConfig): A configuration object.
    """
    if cfg.track:
        # Perform the tracking
        track = Track(cfg)
        assignations = track.run()
        out_path = utils.get_run_path("assignations.json")
        logger.info(f"Saving assignations to: {out_path}")
        save_assignations(out_path, assignations)
    else:
        # Load assignations from file
        logger.info(f"Loading assignations from: {cfg.assignations_path}")
        assignations = load_assignations(cfg.assignations_path)

    # Run tasks
    for task_params in cfg.tasks:
        logger.info(f"Running task: {task_params._target_}")
        task = instantiate(task_params, cfg=cfg, _recursive_=False)
        task.run()


if __name__ == "__main__":
    main()
