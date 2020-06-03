#
#   Copyright EAVISE
#   Author: Maarten Vandersteegen
#

"""
Dollar
------
"""
import logging
from .annotation import *

__all__ = ["DollarAnnotation", "DollarParser"]
log = logging.getLogger(__name__)


class DollarAnnotation(Annotation):
    """ Dollar image annotation """

    def serialize(self):
        """ generate a dollar annotation string """
        string = "{} {} {} {} {} {} {} {} {} {} {} 0" \
            .format(self.class_label if len(self.class_label) != 0 else '?',
                    round(self.x_top_left),
                    round(self.y_top_left),
                    round(self.width),
                    round(self.height),
                    int(self.occluded),
                    round(self.visible_x_top_left),
                    round(self.visible_y_top_left),
                    round(self.visible_width),
                    round(self.visible_height),
                    int(self.lost))

        return string

    def deserialize(self, string, occlusion_tag_map):
        """ parse a dollar annotation string """
        elements = string.split()
        self.class_label = '' if elements[0] == '?' else elements[0]
        self.x_top_left = float(elements[1])
        self.y_top_left = float(elements[2])
        self.width = float(elements[3])
        self.height = float(elements[4])
        if occlusion_tag_map is None:
            self.occluded = elements[5] != '0'
        else:
            self.occluded_fraction = occlusion_tag_map[int(elements[5])]
        self.visible_x_top_left = float(elements[6])
        self.visible_y_top_left = float(elements[7])
        self.visible_width = float(elements[8])
        self.visible_height = float(elements[9])
        self.lost = elements[10] != '0'

        self.object_id = 0

        return self


class DollarParser(Parser):
    """
    This parser is designed to parse the version3 text based dollar annotation format from Piotr Dollar's MATLAB toolbox_.

    Keyword Args:
        occlusion_tag_map (list, optional): When the occluded flag in the dollar text file (see below) is used as an occlusion level tag, \
        its value is used as an index on this list to obtain an occlusion fraction that will be stored in the ``occluded_fraction`` attribute.

    The dollar format has one .txt file for every image of the dataset where each line within a file represents a bounding box.
    Each line is a space separated list of values structured as follows:

        <label> <x> <y> <w> <h> <occluded> <vx> <vy> <vw> <vh> <ignore> <angle>

    ========  ===========
    Name      Description
    ========  ===========
    label     class label name (string)
    x         left top x coordinate of the bounding box in pixels (integer)
    y         left top y coordinate of the bounding box in pixels (integer)
    w         width of the bounding box in pixels (integer)
    h         height of the bounding box in pixels (integer)
    occluded  1 indicating the object is occluded, 0 indicating the object is not occluded
    vx        left top x coordinate of the inner bounding box that frames the non-occluded part of the object (the visible part)
    vy        left top y coordinate of the inner bounding box that frames the non-occluded part of the object (the visible part)
    vw        width of the inner bounding box that frames the non-occluded part of the object (the visible part)
    vh        height of the inner bounding box that frames the non-occluded part of the object (the visible part)
    lost      1 indicating the object is no visible in the image, 0 indicating the object is (partially) visible
    angle     [0-360] degrees orientation of the bounding box (currently not used)
    ========  ===========

    Example:
        >>> image_000.txt
            % bbGt version=3
            person 488 232 34 100 0 0 0 0 0 0 0
            person 576 219 27 68 0 0 0 0 0 0 0

    Note:
        if no visible bounding box is annotated, [vx, vy, vw, vh] are equal to 0.

    .. _toolbox: https://github.com/pdollar/toolbox/blob/master/detector/bbGt.m
    """
    parser_type = ParserType.MULTI_FILE
    box_type = DollarAnnotation

    def __init__(self, **kwargs):
        self.occlusion_tag_map = None
        if 'occlusion_tag_map' in kwargs:
            self.occlusion_tag_map = kwargs['occlusion_tag_map']
        else:
            log.info("No 'occlusion_tag_map' kwarg found, interpreting occluded value as a binary label.")

    def deserialize(self, string):
        """ deserialize a string containing the content of a dollar .txt file

        This deserializer checks for header/comment strings in dollar strings
        """
        result = []

        for line in string.splitlines():
            if '%' not in line:
                anno = self.box_type()
                result += [anno.deserialize(line, self.occlusion_tag_map)]

        return result
