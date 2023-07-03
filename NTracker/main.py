
import hydra
from omegaconf import DictConfig



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Run the tracker tasks.

    Args:
        cfg (DictConfig): A configuration object.
    """
    


if __name__ == "__main__":
    main()