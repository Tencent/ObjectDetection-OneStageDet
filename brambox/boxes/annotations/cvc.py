#
#   Copyright EAVISE
#   Author: Maarten Vandersteegen
#
"""
CVC
---
"""

from .annotation import *

__all__ = ["CvcAnnotation", "CvcParser"]


class CvcAnnotation(Annotation):
    """ Cvc image annotation """

    def serialize(self):
        """ generate a cvc annotation string

        Note that this format does not support a class label
        """
        string = "{} {} {} {} 1 0 0 0 0 {} 0" \
            .format(round(self.x_top_left + self.width / 2),
                    round(self.y_top_left + self.height / 2),
                    round(self.width),
                    round(self.height),
                    int(self.object_id))

        return string

    def deserialize(self, string):
        """ parse a cvc annotation string

        x,y are the center of a box
        """
        elements = string.split()
        self.width = float(elements[2])
        self.height = float(elements[3])
        self.x_top_left = float(elements[0]) - self.width / 2
        self.y_top_left = float(elements[1]) - self.height / 2
        self.object_id = int(elements[9])

        self.lost = False
        self.occluded = False


class CvcParser(Parser):
    """
    This parser is designed to parse the CVC_ pedestrian dataset collection.
    The CVC format has one .txt file for every image of the dataset where each line within a file represents a bounding box.
    Each line is a space separated list of values structured as follows:

        <x> <y> <w> <h> <mandatory> <unknown> <unknown> <unknown> <unknown> <track_id> <unknown>

    =========  ===========
    Name       Description
    =========  ===========
    x          center x coordinate of the bounding box in pixels (integer)
    y          center y coordinate of the bounding box in pixels (integer)
    w          width of the bounding box in pixels (integer)
    h          height of the bounding box in pixels (integer)
    mandatory  1 if the pedestrian is mandatory for training and testing, 0 for optional
    track_id   identifier of the track this object is following (integer)
    =========  ===========

    Example:
        >>> image_000.txt
            97 101 18 52 1 0 0 0 0 1 0
            121 105 15 46 1 0 0 0 0 2 0
            505 99 14 41 1 0 0 0 0 3 0

    Warning:
        This parser is only tested on the CVC-14 dataset

    .. _CVC: http://adas.cvc.uab.es/elektra/datasets/pedestrian-detection/
    """
    parser_type = ParserType.MULTI_FILE
    box_type = CvcAnnotation
