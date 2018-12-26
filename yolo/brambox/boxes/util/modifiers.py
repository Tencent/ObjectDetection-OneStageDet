#
#   Copyright EAVISE
#   Author: Maarten Vandersteegen
#   Author: Tanguy Ophoff
#
"""
These modifier functions allow to change certain aspects of your annotations and detections.
"""
import collections
from ..annotations import Annotation


def modify(boxes, modifier_fns):
    """ Modifies boxes according to the modifier functions.

    Args:
        boxes (dict or list): Dictionary containing box objects per image ``{"image_id": [box, box, ...], ...}`` or list of bounding boxes
        modifier_fns (list): List of modifier functions that get applied

    Returns:
        (dict or list): boxes after modifications

    Warning:
        These modifier functions will mutate your bounding boxes and some of them can even remove bounding boxes.
        If you want to keep a copy of your original values, you should pass a copy of your bounding box dictionary:

        >>> import copy
        >>> import brambox.boxes as bbb
        >>>
        >>> new_boxes = bbb.modify(copy.deepcopy(boxes), [modfier_fns, ...])
    """

    if isinstance(boxes, dict):
        for _, values in boxes.items():
            for i in range(len(values)-1, -1, -1):
                for fn in modifier_fns:
                    values[i] = fn(values[i])
                    if values[i] is None:
                        del values[i]
                        break
    else:
        for i in range(len(boxes)-1, -1, -1):
            for fn in modifier_fns:
                boxes[i] = fn(boxes[i])
                if boxes[i] is None:
                    del boxes[i]
                    break

    return boxes


class AspectRatioModifier:
    """ Change the aspect ratio of all bounding boxes in ``boxes``.

    Args:
        aspect_ratio (Number, optional): Target aspect ratio, defined as height/width; Default **1.0**
        change (str, optional): which length to change; Default **'width'**

    Note:
        The ``change`` parameter can be one of 4 different values.
        If the parameter is **'width'**, then the width of the bounding box will be modified to reach the new aspect ratio.
        If it is **'height'**, then the height of the bounding box will be modified. |br|
        If the parameter is **'reduce'**, then the bounding box will be cropped to reach the new aspect ratio.
        If it is **'enlarge'**, then the bounding box will be made bigger.
    """
    def __init__(self, aspect_ratio=1.0, change='width', modify_ignores=False):
        self.ar = aspect_ratio
        self.modify_ignores = modify_ignores
        change = change.lower()
        if change == 'reduce':
            self.change = 0
        elif change == 'enlarge':
            self.change = 1
        elif change == 'height':
            self.change = 2
        else:
            self.change = 3

    def __call__(self, box):
        if not self.modify_ignores and hasattr(box, 'ignore') and box.ignore:
            return box

        change = False
        if self.change == 0:
            if box.height / box.width > self.ar:
                change = True
        elif self.change == 1:
            if box.height / box.width < self.ar:
                change = True

        if self.change == 2 or change:
            d = box.width * self.ar - box.height
            box.y_top_left -= d / 2
            box.height += d
        else:
            d = box.height * self.ar - box.width
            box.x_top_left -= d / 2
            box.width += d

        return box


class ScaleModifier:
    """ Rescale your bounding boxes like you would rescale an image.

    Args:
        scale (Number or list, optional): Value to rescale your bounding box, defined as a single number or a (width, height) tuple; Default **1.0**
    """
    def __init__(self, scale=1.0):
        if isinstance(scale, collections.Sequence):
            self.scale = tuple(scale[:2])
        else:
            self.scale = (scale, scale)

    def __call__(self, box):
        box.x_top_left *= self.scale[0]
        box.y_top_left *= self.scale[1]
        box.width *= self.scale[0]
        box.height *= self.scale[1]

        return box


class CropModifier:
    """ Crop bounding boxes to fit inside a certain area.

    Args:
        area (number or list, optional): area that your bounding box should be cropped in, defined as a (x, y, w, h) tuple; Default **(0, 0, Inf, Inf)**
        intersection_threshold (number or list, optional): Fraction of the bounding box that should still be inside the cropped ``area``; Default **0**
        move_origin (boolean, optional): This value indicates whether we should move the origin of the coordinate system to the top-left corner of the cropped ``area``; Default **True**
        discard_lost (boolean, optional): Whether to discard bounding boxes that are not in the ``area`` or just set the ``lost`` flag to **True**; Default **True**
        update_truncated (boolean, optional): *!For annotations only!* Update the ``truncated_fraction`` property if necessary; Default **False**

    Note:
        The ``area`` parameter can have multiple type of values. |br|
        If a list of 4 values is given, it is interpreted as an area with **(x, y, w, h)**.
        If there are only 3 values, they are interpreted as a square area with **(x, y, size)**.
        If you pass only 2 values, it will be interpreted as a square area with the same x and y starting position.
        You can also pass a single number, which will then be interpreted as a square area that starts at position (0,0).

    Note:
        The ``intersection_threshold`` parameter can have multiple type of values. |br|
        If you use a single value then the decision to keep a bounding box will be made according to the following formula:
        :math:`\\frac {area_{box\\ in\\ cropped\\ area}} {area_{box}} >= intersection\\_threshold`

        If you use a **(width_thresh, height_thresh)** tuple, then the following formula is used:
        :math:`\\frac {width_{box\\ in\\ cropped\\ area}} {width_{box}} \\geq width\\_thresh \\ \\& \\  \\frac {height_{box\\ in\\ cropped\\ area}} {height_{box}} \\geq height\\_thresh`
    """
    def __init__(self, area=float('Inf'), intersection_threshold=0, move_origin=True, discard_lost=True, update_truncated=False):
        if isinstance(area, collections.Sequence):
            if len(area) >= 4:
                self.area = tuple(area[:4])
            elif len(area) == 3:
                self.area = (area[0], area[1], area[2], area[2])
            elif len(area) == 2:
                self.area = (area[0], area[0], area[1], area[1])
            else:
                self.area = (0, 0, area[0], area[0])
        else:
            self.area = (0, 0, area, area)
        self.area = (self.area[0], self.area[1], self.area[0] + self.area[2], self.area[1] + self.area[3])

        if isinstance(intersection_threshold, collections.Sequence):
            self.inter_thresh = tuple(intersection_threshold[:2])
            self.inter_area = False
        else:
            self.inter_thresh = intersection_threshold
            self.inter_area = True

        self.move_origin = move_origin
        self.discard_lost = discard_lost
        self.update_truncated = update_truncated

    def __call__(self, box):
        x1 = max(self.area[0], box.x_top_left)
        y1 = max(self.area[1], box.y_top_left)
        x2 = min(self.area[2], box.x_top_left+box.width)
        y2 = min(self.area[3], box.y_top_left+box.height)
        w = x2-x1
        h = y2-y1

        #print(x1, x2, w, h)
        if self.inter_area:
            ratio = ((w * h) / (box.width * box.height)) < self.inter_thresh
        else:
            ratio = (w / box.width) < self.inter_thresh[0] or (h / box.height) < self.inter_thresh[1]
        if w <= 0 or h <= 0 or ratio:
            if self.discard_lost:
                #print(w, h)
                return None
            else:
                box.lost = True
                if self.update_truncated and isinstance(box, Annotation) and box.truncated_fraction < 1:
                    if w <= 0 or h <= 0:
                        box.truncated_fraction = 1
                    else:
                        box.truncated_fraction = max(0, 1 - ((w * h) / (box.width * box.height * 1/(1-box.truncated_fraction))))
                if self.move_origin:
                    box.x_top_left -= self.area[0]
                    box.y_top_left -= self.area[1]
                return box
        else:
            if self.update_truncated and isinstance(box, Annotation) and box.truncated_fraction < 1:
                box.truncated_fraction = max(0, 1 - ((w * h) / (box.width * box.height * 1/(1-box.truncated_fraction))))
            box.x_top_left = x1
            box.y_top_left = y1
            box.width = w
            box.height = h

            if self.move_origin:
                box.x_top_left -= self.area[0]
                box.y_top_left -= self.area[1]

            #print(box.x_top_left, box.y_top_left)
            return box
