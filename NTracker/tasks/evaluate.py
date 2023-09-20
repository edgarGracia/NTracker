import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

from omegaconf import DictConfig

from NTracker.utils.path_utils import get_run_path
from NTracker.utils.tracking_utils import load_track

logger = logging.getLogger(__name__)


def almost_equal(a: float, b: float, threshold: float) -> bool:
    if b <= a+threshold and b >= a-threshold:
        return True
    return False


def almost_equal_pos(pos_a: dict, pos_b: dict, threshold: float)  -> bool:
    return (
        almost_equal(pos_a["x"], pos_b["x"], threshold) and
        almost_equal(pos_a["y"], pos_b["y"], threshold)
    )


class Evaluate:
    """Evaluate the assignations in two different frames using a ground truth.
    """

    def __init__(
        self,
        cfg: DictConfig,
        first_frame: int,
        last_frame: int,
        ground_truth: Union[Path, str],
        pos_threshold: int = 5,
        output_file: Optional[Union[Path, str]] = "evaluation.json",
    ):
        """Create a tracking evaluation object.

        Args:
            cfg (DictConfig): A configuration object.
            first_frame (int): First frame to evaluate.
            last_frame (int): Last frame to evaluate.
            pos_threshold (int, optional): Position threshold to match the
                predicted instances with the ground truth. Defaults to 1.
            output_file (Optional[Union[Path, str]], optional): Output file
                path. Relative paths are appended to the run path. Defaults to
                "evaluation.json"
        """
        output_file = Path(output_file)

        self.cfg = cfg
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.ground_truth = Path(ground_truth)
        self.pos_threshold = pos_threshold
        self.output_file = (
            output_file if output_file.is_absolute() else get_run_path(output_file)
        )

    def run(self, tracking_data: Dict[int, Dict[int, Dict[str, int]]]):
        """Run the evaluation.

        Args:
            tracking_data (Dict[int, Dict[int, Dict[str, int]]]): Tracking data:
                ({tracked_id: {frame_n: {original_id: , x: ..., y: ...}}})
        """
        # Read the ground truth
        gt_data = load_track(self.ground_truth)

        # Compute last frame id translation dict
        translate_id_first = dict()
        for g_k, g_frames in gt_data.items():
            assert self.first_frame in g_frames, (
                f"Last frame {self.first_frame} not in ground truth "
                f"for instance id {g_k}"
            )
            translate_id_first[g_k] = None
            for t_k, t_frames in tracking_data.items():
                if (
                    self.first_frame in t_frames
                    and almost_equal_pos(
                        g_frames[self.first_frame],
                        t_frames[self.first_frame],
                        self.pos_threshold
                    )
                ):
                    assert translate_id_first[g_k] is None, (
                        f"Instance with duplicated position found ({t_k}). "
                        f"Ground truth ID {g_k} already assigned to "
                        f"{translate_id_first[g_k]} at frame {self.first_frame}"
                    )
                    translate_id_first[g_k] = t_k

        # Compute last frame id translation dict
        translate_id_last = dict()
        for g_k, g_frames in gt_data.items():
            assert self.last_frame in g_frames, (
                f"Last frame {self.last_frame} not in ground truth "
                f"for instance id {g_k}"
            )
            translate_id_last[g_k] = None
            for t_k, t_frames in tracking_data.items():
                if (
                    self.last_frame in t_frames
                    and almost_equal_pos(
                        g_frames[self.last_frame],
                        t_frames[self.last_frame],
                        self.pos_threshold
                    )
                ):
                    assert translate_id_last[g_k] is None, (
                        f"Instance with duplicated position found ({t_k}). "
                        f"Ground truth ID {g_k} already assigned to "
                        f"{translate_id_last[g_k]} at frame {self.last_frame}"
                    )
                    translate_id_last[g_k] = t_k

        assert sorted(list(translate_id_first.keys())) == \
            sorted(list(translate_id_last.keys())), (
                f"Ground truth keys mismatch at first ({self.first_frame}) and "
                f"last ({self.last_frame}) frames: "
                f"{sorted(list(translate_id_first.keys()))} "
                f"{sorted(list(translate_id_last.keys()))}"
            )

        incorrect_gt_id = []
        incorrect_track_id = []
        for g_k in translate_id_first.keys():
            if (translate_id_first[g_k] != translate_id_last[g_k]
                or translate_id_first[g_k] is None):
                incorrect_gt_id.append(g_k)
                incorrect_track_id.append(translate_id_last[g_k])

        num_id = len(translate_id_first)
        corrects_num = num_id - len(incorrect_gt_id)
        corrects_rel = corrects_num / num_id

        result_str = f"Tracking evaluation on frames {self.first_frame} and "
        result_str += f"{self.last_frame}\n"
        result_str += f"{corrects_num} of {num_id} ({corrects_rel})\n"
        result_str += f"Wrong ground truth IDs: {incorrect_gt_id}\n"
        result_str += f"Wrong tracked IDs: {incorrect_track_id}"

        eval_data = {
            "incorrect_gt_id": incorrect_gt_id,
            "incorrect_track_id": incorrect_track_id,
            "num_id": num_id,
            "corrects_num": corrects_num,
            "corrects_rel": corrects_rel,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "str": result_str
        }
        
        logger.info(result_str)
        logger.info(f"Saving evaluation to {self.output_file}")
        self.output_file.write_text(json.dumps(eval_data, indent=4))
