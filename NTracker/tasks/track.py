
import hydra
from omegaconf import DictConfig
from pathlib import Path

from tqdm import tqdm
import cv2
from NTracker.tracking.tracker import Tracker
from NTracker.utils import utils

class Track:

    def __ini__(self, cfg: DictConfig):
        self.cfg = cfg
    
    def run(self):

        images_path = Path(self.cfg.images_path)
        load_images = self.cfg.tracker.load_images

        
        tracker = Tracker(self.cfg)

        # List input images
        images_paths = [i for i in images_path.iterdir()
                        if i.suffix.lower() in self.cfg.images_extensions
        ]
        images_paths = utils.sort_numerical_paths(images_paths)
        
        # Set start and end frames
        start_frame = (self.cfg.start_frame
                       if self.cfg.start_frame is not None else 0)
        end_frame = (self.cfg.end_frame
                     if self.cfg.end_frame is not None else len(images_path))

    # init_instances = config["tracking"]["init_instances"]
    # end_frame = len(images_paths) if end_frame < 0 else end_frame
        
        try:
            for img_i, img_path in enumerate(
                tqdm(images_path[start_frame:end_frame])
            ):
                # Read image
                if load_images:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        raise IOError(f"Can not read image {str(img_path)}")
                else:
                    img = None
                
                # Read annotation
                coco = coco_path.joinpath(img_path.stem + ".json")
                assert coco.exists(), coco
                instances = utils.read_coco(coco)
                instances = {i: x for i, x in enumerate(instances)}

                # Filter num instances
                n_ins = config["tracking"]["num_instances"]
                if (config["tracking"]["filter_frames"] and
                    n_ins != len(instances)):
                    continue
                
                if init_instances is not None:
                    if init_instances != len(instances):
                        continue
                    else:
                        init_instances = None
                
                if config["tracking"]["filter_instances"]:
                    if n_ins != len(instances):
                        i_sort = sorted(
                            instances.items(),
                            key=lambda x: x[1]["score"],
                            reverse=True
                        )
                        instances = {
                            i[0]: i[1] for i in i_sort[:n_ins]
                        }

                # Track
                tracking.reset()
                for i, ins in instances.items():
                    tracking.add_instance(img, ins, i, img_path)
                instances = tracking.re_assign(instances)

                # Call tasks
                out_img = img
                for t in tasks:
                    out_img = t.set(img_id, img_path, out_img, instances)
                image_saver.set_image(img_id, out_img)

        except KeyboardInterrupt:
            pass

    task_results = {t: t.end() for t in tasks}
    image_saver.close()
    return task_results