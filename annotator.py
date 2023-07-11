import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import PySimpleGUI as sg
from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf
from hydra.utils import instantiate
from NTracker.utils import path_utils, image_utils


from NTracker.visualization import draw




class AnnotatorGui:
        
    SG_IMAGE = "SG_IMAGE"
    SG_ANNOTS_LIST = "SG_ANNOTS_LIST"
    BUTTON_SAVE = "BUTTON_SAVE"
    IMAGE_LIST_WIDTH = 50

    def __init__(
        self,
        images_path: Path,
        annotations_path: Path,
        output_path: Path,
        config_path: Path
    ):
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.output_path = output_path
        self.window_size = (0,0)
        self.image = None
        
        # Get the cfg
        initialize_config_dir(
            config_dir=str(config_path.parent),
            job_name="annotator",
            version_base=None
        )
        self.cfg = compose(config_name=config_path.stem, overrides=[])

        # Create the annotations parser and list files
        self.cfg.annotations_parser.base_path = self.annotations_path
        self.annotations_parser = instantiate(self.cfg.annotations_parser)
        self.annotations_paths = self.annotations_parser.list_annotations()

        self._run()

    def _get_layout(self):
        left_panel = sg.Frame("", vertical_alignment="top", layout=[
            [
                sg.Listbox([], select_mode=sg.LISTBOX_SELECT_MODE_SINGLE,
                           bind_return_key=True, key=self.SG_ANNOTS_LIST,
                           size=(self.IMAGE_LIST_WIDTH,99999)),
            ],
        ])
        center_panel = sg.Frame("", vertical_alignment="top", layout=[
            [
                sg.Column([[
                    sg.Graph((2000,2000), (0,2000), (2000,0), key=self.SG_IMAGE,
                        enable_events=True, expand_x=True, expand_y=True)
                ]], scrollable=True, )
            ],
        ])
        main_layout = [
            [left_panel, sg.Push(), center_panel]
        ]

        return main_layout

    def _init_fields(self):
        self.window_size = self.window.size
        self.window[self.SG_ANNOTS_LIST].update(self.annotations_paths)

    def show_image(self):
        if self.image is None:
            return
        
        image = self.image
        h, w = image.shape[0:2]
        
        # graph_w = 
        # graph_h = 

        target_w = max(128, min(w, (self.window_size[0] - self.IMAGE_LIST_WIDTH) - 400))
        target_h = max(128, min(h, self.window_size[1]) - 50)
        
        ratio_graph = self.window_size[0]/self.window_size[1]
        ratio_img = w/h

        if ratio_graph <= ratio_img:
            image, _, _ = image_utils.resize_image(image, width=target_w)
        else:
            image, _, _ = image_utils.resize_image(image, height=int(target_h))
        
        img_encode = cv2.imencode('.png', image)[1].tobytes()
        self.window[self.SG_IMAGE].erase()
        self.window[self.SG_IMAGE].draw_image(
            data=img_encode, location=(0,0)
        )

    def set_annotation(self, path: Path):
        instances = self.annotations_parser.read(path)
        image_path = path_utils.get_sibling_path(
            path,
            self.images_path,
            self.cfg.images_extensions
        )[0]
        image = cv2.imread(str(image_path))

        for i, ins in enumerate(instances):
            image = draw.draw_instance(
                self.cfg,
                image,
                0,
                i,
                ins,
                None
            )

        self.image = image
        self.show_image()
        
    def on_resize_window(self):
        self.show_image()
        


        
    def _run(self):
        self.window = sg.Window("Annotator", self._get_layout(), resizable=True,
                                size=(800,800), return_keyboard_events=True)
        self.window.Finalize()
        self.window.bind('<Configure>', "Configure") # To respond to window resize events

        self._init_fields()

        while True:
            event, values = self.window.read()

            # Close window
            if event == sg.WIN_CLOSED:
                break
            
            # Window resized
            elif event == 'Configure':
                if self.window_size != self.window.size:
                    self.window_size = self.window.size
                    self.on_resize_window()
            
            # Select image from list
            elif event == self.SG_ANNOTS_LIST:
                self.set_annotation(values[self.SG_ANNOTS_LIST][0])
            
            # window[self.].set_focus()


        self.window.close()



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GUI application to annotate the "
                                 "position and ID of instances. It generate a "
                                 "json that can be later used to evaluate the "
                                 "performance of the tracker.")
    ap.add_argument(
        "-i",
        "--images",
        help="Images path",
        type=Path,
        required=True
    )
    ap.add_argument(
        "-a",
        "--annotations",
        help="Annotations path",
        type=Path,
        required=True
    )
    ap.add_argument(
        "-o",
        "--output",
        help="Output JSON file. If exists it will append the annotations",
        type=Path,
        required=True
    )
    default_config = Path(__file__).parent / "NTracker/conf/coco_track.yaml"
    ap.add_argument(
        "-c",
        "--config",
        help=f"Path to a configuration YAML. Defaults to {default_config}",
        type=Path,
        default=default_config
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    AnnotatorGui(
        args.images,
        args.annotations,
        args.output,
        args.config
    )
