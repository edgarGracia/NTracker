from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import seaborn as sns
from omegaconf import DictConfig

from NTracker.utils.structures import Instance
from NTracker.visualization import RelativePosition

CV2_FONT = cv2.FONT_HERSHEY_SIMPLEX


def get_color(
    instance_id: int,
    palette: Optional[Union[str, List[Tuple[int, int, int]]]],
) -> Tuple[int, int, int]:
    """Get a color for an instance ID.

    Args:
        instance_id (int): An instance ID.
        palette (Optional[Union[str, List[Tuple[int, int, int]]]]): The name of
            a Seaborn palette or a list of colors.

    Returns:
        Tuple[int, int, int]: A color.
    """
    if isinstance(palette, str) or palette is None:
        return [
            int(i*255) for i in
            sns.color_palette(
                palette, instance_id+1
            )[instance_id]
        ]
    return palette[instance_id]


def get_text_position_from_box(
    box: Tuple[int, int, int, int],
    relative_position: Union[RelativePosition, str]
) -> Tuple[int, int]:
    """Get the correct initial position of a text given a bounding box.

    Args:
        box (Tuple[int, int, int, int]): A bounding box (xmin, ymin, xmax, ymax).
        relative_position (Union[RelativePosition, str]): In which corner of
            the box put the text.

    Returns:
        Tuple[int, int]: Final text position (x, y).
    """
    relative_position = RelativePosition(relative_position)
    xmin, ymin, xmax, ymax = box
    if relative_position is RelativePosition.TOP_LEFT:
        return (xmin, ymin)
    elif relative_position is RelativePosition.TOP_RIGHT:
        return (xmax, ymin)
    elif relative_position is RelativePosition.BOTTOM_LEFT:
        return (xmin, ymax)
    elif relative_position is RelativePosition.BOTTOM_RIGHT:
        return (xmax, ymax)
    elif relative_position is RelativePosition.CENTER:
        return ((xmax-xmin)//2, (ymax-ymin)//2)
    else:
        raise NotImplementedError(str(relative_position))


def draw_text(
    image: np.ndarray,
    text: Union[str, List[str]],
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    scale: int = 1,
    thickness: int = 1,
    line_space: int = 15,
    relative_position: Union[RelativePosition,
                             str] = RelativePosition.TOP_RIGHT,
    background: bool = True,
    background_color: Tuple[int, int, int] = (50, 50, 50),
    background_alpha: float = 1,
    margin: int = 0
) -> np.ndarray:
    """Draw text in the given image.

    Args:
        image (np.ndarray): The source image where draw the text.
        text (Union[str, List[str]]): The text to draw. A list of lines or a
            single string. The line breaks ('\n') will be parsed to lines.
        position (Tuple[int, int]): The text position within the image (x, y).
        color (Tuple[int, int, int], optional): The color of the text.
            Defaults to (255, 255, 255).
        scale (int, optional): The scale of the text. Defaults to 1.
        thickness (int, optional): The text thickness_. Defaults to 1.
        line_space (int, optional): Line spacing. Defaults to 15.
        relative_position (Union[RelativePosition, str], optional): Relative
            position of the text to the provided position. Defaults to
            RelativePosition.TOP_RIGHT.
        background (bool, optional): Whether draw a background area behind the
            actual text. Defaults to True.
        background_color (Tuple[int, int, int], optional): Color of the
            background. Defaults to (50, 50, 50).
        background_alpha (float, optional): Opacity of the background.
            Defaults to 1.
        margin (int, optional): Text margin. Defaults to 0.

    Returns:
        np.ndarray: The source image with the text drawn.
    """
    if not text:
        return image

    relative_position = RelativePosition(relative_position)
    text_lines = text.splitlines() if isinstance(text, str) else text

    # Compute the final text size
    max_text_len = max(text_lines, key=lambda x: len(x))
    (text_w, text_h), _ = cv2.getTextSize(
        max_text_len,
        CV2_FONT,
        scale,
        thickness
    )
    box_h = text_h + ((text_h + scale * line_space) *
                      (len(text_lines) - 1)) + (margin * 2)

    # Compute the text starting position
    x, y = position
    if relative_position is RelativePosition.TOP_RIGHT:
        pass
    elif relative_position is RelativePosition.TOP_LEFT:
        x -= text_w + (margin*2)
    elif relative_position is RelativePosition.BOTTOM_RIGHT:
        y += box_h
    elif relative_position is RelativePosition.BOTTOM_LEFT:
        x -= text_w + (margin*2)
        y += box_h
    else:
        raise NotImplementedError(str(relative_position))

    # Clip values
    x = min(max(x, 0), image.shape[1] - 1)
    y = min(max(y, 0), image.shape[0] - 1)

    # Draw background
    if background and background_alpha > 0:
        xmin = max(x, 0)
        ymin = max(y - box_h, 0)
        xmax = min(x + text_w + (margin * 2), image.shape[1])
        ymax = min(y, image.shape[0])

        if background_alpha < 1:
            bg = np.full(
                (ymax-ymin, xmax-xmin, 3),
                background_color,
                dtype="uint8"
            )
            image[ymin:ymax, xmin:xmax, :] = cv2.addWeighted(
                image[ymin:ymax, xmin:xmax, :],
                1-background_alpha,
                bg,
                background_alpha,
                0
            )
        else:
            cv2.rectangle(
                image,
                (xmin, ymin),
                (xmax, ymax),
                background_alpha,
                -1
            )

    # Draw the text lines
    for i, line in enumerate(reversed(text_lines)):
        dy = i * (text_h + scale * line_space)
        cv2.putText(
            image, line, (x + margin, y - margin - dy), CV2_FONT, scale, color,
            thickness, cv2.LINE_AA)

    return image


def draw_bounding_box(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    alpha: float = 1,
    thickness: int = 1,
    fill: bool = False
) -> np.ndarray:
    """Draw a bounding box in an image.

    Args:
        image (np.ndarray): The source image where draw the text.
        box (Tuple[int, int, int, int]): A bounding box (xmin, ymin, xmax, ymax).
        color (Tuple[int, int, int], optional): Color of the box.
            Defaults to (255, 255, 255).
        alpha (float, optional): Opacity of the box. Defaults to 1.
        thickness (int, optional): Thickness of the box. Defaults to 1.
        fill (bool, optional): Whether to fill the box. Defaults to False.

    Returns:
        np.ndarray: The source image with the box drawn.
    """
    thickness = -1 if fill else thickness

    if alpha <= 0:
        return image

    xmin, ymin, xmax, ymax = box

    if alpha < 1:
        box_mask = cv2.rectangle(
            np.zeros_like(image),
            (xmin, ymin),
            (xmax, ymax),
            (1, 1, 1),
            thickness
        )
        color_mask = np.clip(box_mask * color, 0, 255).astype("uint8")
        image[box_mask == 1] = cv2.addWeighted(
            image[box_mask == 1],
            1-alpha,
            color_mask[box_mask == 1],
            alpha,
            0
        )[:, 0]
    else:
        image = cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            color,
            thickness
        )

    return image


def instance_text_formatter(
    instance_key: int,
    instance: Instance,
    expression: str
) -> str:
    """Create the text to display from an instance.

    WARNING: This function uses the ``eval`` function and can execute arbitrary
    code.

    Args:
        instance_key (int): The key of the instance.
        instance (Instance): An Instance object.
        expression (str): Expression to evaluate that should return the string
            to show along with each instance.

    Returns:
        str: The text with the instance values to show.
    """
    return eval(f"str({expression})")


def draw_instance(
    cfg: DictConfig,
    image: np.ndarray,
    instances: Dict[int, Instance]
) -> np.ndarray:
    """Draw an instance object over an image.

    Args:
        cfg (DictConfig): A configuration object.
        image (np.ndarray): Numpy image of shape (H, W, 3) and "uint8" dtype.
        instances (Dict[int, Instance]): Dict of instances.

    Returns:
        np.ndarray: The image with the instance's data data drawn.
    """
    if cfg.visualization.img_background:
        image = image.copy()
    else:
        image = np.full_like(image, cfg.visualization.img_bg_color)

    for instance_key, instance in instances.items():
        box = instance.bounding_box

        # Draw bounding box
        if cfg.visualization.box.visible:
            if cfg.visualization.box.color_by_id:
                box_color = get_color(
                    instance_id=instance_key,
                    palette=cfg.visualization.box.palette,
                )
            else:
                box_color = cfg.visualization.box.color
            draw_bounding_box(
                image=image,
                box=box,
                color=box_color,
                alpha=cfg.visualization.box.alpha,
                thickness=cfg.visualization.box.thickness,
                fill=cfg.visualization.box.fill
            )

        # Draw text along the box
        if cfg.visualization.text.visible and cfg.visualization.text.formatter:
            text = instance_text_formatter(
                instance_key,
                instance,
                cfg.visualization.text.formatter
            )
            position = get_text_position_from_box(
                box=box,
                relative_position=cfg.visualization.box.text_position
            )
            if cfg.visualization.text.color_by_id:
                text_color = get_color(
                    instance_id=instance_key,
                    palette=cfg.visualization.text.palette
                )
            else:
                text_color = cfg.visualization.text.color
            if cfg.visualization.text_bg.color_by_id:
                text_bg_color = get_color(
                    instance_id=instance_key,
                    palette=cfg.visualization.text_bg.palette,
                )
            else:
                text_bg_color = cfg.visualization.text_bg.color
            draw_text(
                image=image,
                text=text,
                position=position,
                color=text_color,
                scale=cfg.visualization.text.scale,
                thickness=cfg.visualization.text.thickness,
                line_space=cfg.visualization.text.line_space,
                relative_position=cfg.visualization.text.position,
                background=cfg.visualization.text_bg.visible,
                background_color=text_bg_color,
                background_alpha=cfg.visualization.text_bg.alpha,
                margin=cfg.visualization.text_bg.margin
            )
    return image
