import argparse
import json
from pathlib import Path
import math

import cv2
import numpy as np
import PySimpleGUI as sg
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from NTracker.utils import path_utils


from NTracker.visualization import draw
from NTracker.utils import structures


class AnnotatorGui:

    SG_IMAGE = "SG_IMAGE"
    SG_ANNOTS_LIST = "SG_ANNOTS_LIST"
    SG_INPUT_ID = "SG_INPUT_ID"
    SG_INPUT_FRAME = "SG_INPUT_FRAME"
    BUTTON_SAVE = "BUTTON_SAVE"
    IMAGE_LIST_WIDTH = 50

    def __init__(
        self,
        images_path: Path,
        annotations_path: Path,
        input_tracking_data_path: Path,
        output_tracking_data_path: Path,
        config_path: Path
    ):
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.input_tracking_data_path = input_tracking_data_path
        self.output_tracking_data_path = output_tracking_data_path

        self.window_size = (0, 0)
        self.image = None
        self.image_resize_factor = 1
        self.selected_id = None
        self.selected_frame_i = 0

        self.instances = {}

        # Get the cfg
        initialize_config_dir(
            config_dir=str(config_path.absolute().parent),
            job_name="annotator",
            version_base=None
        )
        self.cfg = compose(config_name=config_path.stem, overrides=[])

        # Create the annotations parser and list files
        self.cfg.annotations_parser.base_path = self.annotations_path
        self.annotations_parser = instantiate(self.cfg.annotations_parser)
        self.annotations_paths = self.annotations_parser.list_annotations()

        # Load tracking data
        if input_tracking_data_path:
            self.tracking_data = json.loads(
                input_tracking_data_path.read_text(),
                object_hook=lambda d: {
                    int(k) if k.isdigit() else k: v for k, v in d.items()}
            )
        else:
            self.tracking_data = {}

        self.run()

    def get_layout(self):
        left_panel = sg.Frame("", vertical_alignment="top", layout=[
            [
                sg.Text("ID:"),
                sg.Input("", size=(self.IMAGE_LIST_WIDTH-3, 1),
                         key=self.SG_INPUT_ID, enable_events=True)
            ],
            [
                sg.Text("Frame:"),
                sg.Input("", size=(self.IMAGE_LIST_WIDTH-7, 1),
                         disabled=True, key=self.SG_INPUT_FRAME)
            ],
            [
                sg.Listbox([], select_mode=sg.LISTBOX_SELECT_MODE_SINGLE,
                           bind_return_key=True, key=self.SG_ANNOTS_LIST,
                           size=(self.IMAGE_LIST_WIDTH, 99999)),
            ],
        ])
        center_panel = sg.Frame("", vertical_alignment="top", layout=[
            [
                sg.Column([[
                    sg.Graph((2000, 2000), (0, 2000), (2000, 0),
                             key=self.SG_IMAGE, enable_events=True)
                ]], scrollable=True, )
            ],
        ])
        main_layout = [
            [left_panel, sg.Push(), center_panel]
        ]

        return main_layout

    def init_fields(self):
        self.window_size = self.window.size
        self.window[self.SG_ANNOTS_LIST].update(self.annotations_paths)

    def show_image(self):
        if self.image is None:
            return

        vis_img = self.draw_instances()
        image_h, image_w = vis_img.shape[0:2]

        graph_w = max(1, self.window_size[0] - self.IMAGE_LIST_WIDTH - 400)
        graph_h = max(1, self.window_size[1] - 50)

        ratio_img = image_w / image_h
        ratio_graph = graph_w / graph_h

        if ratio_img > ratio_graph:
            f = graph_w / image_w
        else:
            f = graph_h / image_h

        vis_img = cv2.resize(vis_img, None, fx=f, fy=f)
        self.image_resize_factor = f

        img_encode = cv2.imencode('.png', vis_img)[1].tobytes()
        self.window[self.SG_IMAGE].erase()
        self.window[self.SG_IMAGE].draw_image(
            data=img_encode, location=(0, 0)
        )

    def draw_instances(self) -> np.ndarray:
        vis_img = self.image.copy()
        for k, ins in self.instances.items():
            vis_img = draw.draw_instance(
                self.cfg,
                vis_img,
                self.selected_frame_i,
                k,
                ins,
                {}
            )
        return vis_img

    def set_annotation(self, path: Path):
        self.selected_frame_i = self.annotations_paths.index(path)
        self.window[self.SG_INPUT_FRAME].update(self.selected_frame_i)
        all_instances = self.annotations_parser.read(path)
        all_instances = {int(ins.id): ins for ins in all_instances}
        self.instances = {}
        for k, frames in self.tracking_data.items():
            if self.selected_frame_i in frames:
                self.instances[k] = all_instances[
                    frames[self.selected_frame_i]["original_id"]]

        if not self.instances:
            self.instances = all_instances
        else:
            not_seen_ins = set([ins.id for ins in all_instances.values()]) - \
                set([ins.id for ins in self.instances.values()])

            for ns in not_seen_ins:
                k_set = set(self.instances.keys())
                new_k = sorted(list(set(range(max(k_set)+2)) - k_set))[0]
                self.instances[new_k] = all_instances[ns]

        image_path = path_utils.get_sibling_path(
            path,
            self.images_path,
            self.cfg.images_extensions
        )[0]
        image = cv2.imread(str(image_path))
        self.image = image
        self.show_image()

    def find_nearest_instance(self, position) -> int:
        x, y = position
        x /= self.image_resize_factor
        y /= self.image_resize_factor
        min_k, min_dist = None, float("inf")
        for k, ins in self.instances.items():
            ix, iy = structures.box_center(ins.bounding_box)
            dist = math.sqrt((ix - x)**2 + (iy - y)**2)
            if dist < min_dist:
                min_dist = dist
                min_k = k
        return min_k

    def on_graph_click(self, position):
        self.selected_id = self.find_nearest_instance(position)
        self.window[self.SG_INPUT_ID].update(self.selected_id)
        self.window[self.SG_INPUT_ID].set_focus()

    def on_resize_window(self):
        self.show_image()

    def update_instance_key(self, new_key: str):
        try:
            new_key = int(new_key)
        except:
            return
        if self.selected_id is None or self.selected_id not in self.instances:
            return

        tmp_ins = self.instances[new_key] if new_key in self.instances else None
        self.instances[new_key] = self.instances[self.selected_id]
        del (self.instances[self.selected_id])
        if tmp_ins is not None:
            self.instances[self.selected_id] = tmp_ins

        self.selected_id = new_key
        self.show_image()
        self.update_tracking_data()
        self.save_tracking_data()

    def update_tracking_data(self):
        for k, ins in self.instances.items():
            if k not in self.tracking_data:
                self.tracking_data[k] = {}
            cx, cy = structures.box_center(ins.bounding_box)
            self.tracking_data[k][self.selected_frame_i] = {
                "original_id": ins.id,
                "x": cx,
                "y": cy
            }

    def save_tracking_data(self):
        # Check deleted IDs
        to_delete = []
        for k, frames in self.tracking_data.items():
            if self.selected_frame_i in frames:
                if k not in self.instances:
                    del (self.tracking_data[k][self.selected_frame_i])
                    if not self.tracking_data[k]:
                        to_delete.append(k)
        [self.tracking_data.pop(k) for k in to_delete]

        self.output_tracking_data_path.write_text(
            json.dumps(self.tracking_data, indent=4))

    def run(self):
        self.window = sg.Window("Tracking Annotator", self.get_layout(),
                                resizable=True, size=(1100, 800),
                                )
        self.window.Finalize()
        # To respond to window resize events
        self.window.bind('<Configure>', "Configure")

        self.init_fields()

        while True:
            event, values = self.window.read()
            # print(event, values)

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
            # Click on the image
            elif event == self.SG_IMAGE:
                self.on_graph_click(values[self.SG_IMAGE])
            elif event == self.SG_INPUT_ID:
                self.update_instance_key(values[self.SG_INPUT_ID])

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
        help="Output tracking data JSON file",
        type=Path,
        required=True
    )
    ap.add_argument(
        "--input",
        help="Input tracking data file, generated from a tracking job or with "
             "the annotator itself.",
        type=Path,
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
        args.input,
        args.output,
        args.config
    )
