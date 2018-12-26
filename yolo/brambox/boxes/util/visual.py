#
#   Copyright EAVISE
#   Author: Tanguy Ophoff
#

import logging
log = logging.getLogger(__name__)   # noqa

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    log.debug('OpenCV not installed, always using PIL')
    cv2 = None

from ..annotations import Annotation
from ..detections import Detection

__all__ = ['draw_boxes']

try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 10)
except FileNotFoundError:
    font = ImageFont.load_default()


def draw_boxes(img, boxes, color=None, show_labels=False, faded=None, method=1):
    """ Draws bounding boxes on the image.

    Args:
        img (OpenCV image or PIL image or filename): Image to draw on
        boxes (list): Bounding boxes to draw
        color (dict or list, optional): Color to use for drawing; Default **every label will get its own color, up to 8 labels**
        show_labels (Boolean, optional): Whether or not to print the label names; Default **False**
        faded (function, optional): Function that determines whether we draw an annotation faded or not; Default **None**
        method (draw_boxes.METHOD_CV or draw_boxes.METHOD_PIL, optional): Whether to use OpenCV or Pillow for opening the image (only useful when filename given); Default: **draw_boxes.METHOD_PIL**

    Returns:
        OpenCV or PIL image: Image with bounding boxes drawn

    Note:
        The ``color`` parameter can either be a dictionary or a list containing a single RGB color.
        If it is a dictionary, the keys represent the different class labels to draw
        and the values are the different RGB colors. |br|
        If no ``color`` parameter is given, the function will give every label its own color,
        by selecting colors from a list of 8 different colors.
    """
    default_colors = [
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
        (255, 255, 255),
        (0, 0, 0)
    ]
    if cv2 is None and method == draw_boxes.METHOD_CV:
        raise ImportError('opencv is not installed')

    # Open image
    if isinstance(img, str) or isinstance(img, Path):
        if method == draw_boxes.METHOD_PIL:
            original = Image.open(img)
            img = ImageDraw.Draw(original)
        else:
            img = cv2.imread(img)
    elif isinstance(img, Image.Image):
        original = img
        img = ImageDraw.Draw(original)
        method = draw_boxes.METHOD_PIL
    elif cv2 is not None and isinstance(img, np.ndarray):
        method = draw_boxes.METHOD_CV
    else:
        raise TypeError(f'Unkown image type [{type(img)}]')

    # Draw boxes
    faded = faded if faded is not None else lambda box: False
    label_color = {}
    color_counter = 0
    for box in boxes:
        text = None
        special = False

        # Type specific
        if isinstance(box, Annotation):
            if box.lost:
                continue
            if faded(box):
                special = True
            if show_labels:
                text = box.class_label
        elif isinstance(box, Detection):
            if faded(box):
                special = True
            if show_labels:
                text = f'{box.class_label} {100*box.confidence:.2f}%'
        else:
            continue

        # Color
        if color is None:
            if box.class_label in label_color:
                use_color = label_color[box.class_label]
            else:
                use_color = default_colors[color_counter]
                label_color[box.class_label] = use_color
                color_counter = (color_counter + 1) % len(default_colors)
        elif isinstance(color, dict):
            if box.class_label not in color:
                continue
            else:
                use_color = color[box.class_label]
        else:
            use_color = color

        # Draw
        if method == draw_boxes.METHOD_PIL:
            draw_pil(img, box, use_color, text, special)
        else:
            draw_cv(img, box, use_color, text, special)

    if method == draw_boxes.METHOD_PIL:
        return original
    else:
        return img


def draw_pil(img, box, color, text, special):
    """ Draw a box on the image. """
    pt1 = (int(box.x_top_left), int(box.y_top_left))
    pt2 = (int(box.x_top_left + box.width), int(box.y_top_left))
    pt3 = (int(box.x_top_left + box.width), int(box.y_top_left + box.height))
    pt4 = (int(box.x_top_left), int(box.y_top_left + box.height))
    thickness = 1 if special else 3
    img.line([pt1, pt2, pt3, pt4, pt1], color, thickness)

    if text is not None:
        offset = 13 if special else 15
        img.text((pt1[0], pt1[1]-offset), text, color, font)


def draw_cv(img, box, color, text, special):
    """ Draw a box on the image. """
    color = (color[2], color[1], color[0])
    pt1 = (int(box.x_top_left), int(box.y_top_left))
    pt2 = (int(box.x_top_left + box.width), int(box.y_top_left + box.height))
    thickness = 1 if special else 3
    cv2.rectangle(img, pt1, pt2, color, thickness)

    if text is not None:
        cv2.putText(img, text, (pt1[0], pt1[1]-5), cv2.FONT_HERSHEY_PLAIN, .75, color, 1, cv2.LINE_AA)


draw_boxes.METHOD_CV = 0
draw_boxes.METHOD_PIL = 1
