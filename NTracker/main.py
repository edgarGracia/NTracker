
import hydra
from omegaconf import DictConfig



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """_summary_

    Args:
        cfg (DictConfig): A configuration object.
    """