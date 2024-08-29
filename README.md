# NTracker

A tracker designed for scenarios with a constant number of instances, based on position,  visual features, and mask-IoU. This tracker is meant to be applied to COCO or YOLO annotation files.

![Sample GIF](res/sample.gif)

## Installation

```
pip install -r requirements.txt
pip install -e .
```

## Usage
1. Run an instance segmentation model and save the results in COCO or YOLO formats.
1. Edit the configuration files in ``NTracker/conf``
1. Run ``python main.py --config-name <config_name>`` to run the tracker. <br> Use ``python main.py --help`` for more details.