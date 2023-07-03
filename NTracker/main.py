import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from NTracker.tasks.track import Track

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Run the tracker tasks.

    Args:
        cfg (DictConfig): A configuration object.
    """
    for task_params in cfg.tasks:
        logger.info(f"Running task: {task_params._target_}")
        task = instantiate(task_params, cfg=cfg, _recursive_=False)
        task.run()


if __name__ == "__main__":
    main()