#
#   Copyright EAVISE
#   Author: Maarten Vandersteegen
#
"""
Coco
----
"""

import json
from .detection import *

__all__ = ["CocoDetection", "CocoParser"]


class CocoDetection(Detection):
    """ Json based detection format from darknet framework """

    def serialize(self):
        """ generate a json detection object """

        raise NotImplementedError

    def deserialize(self, json_obj, class_label_map):
        """ parse a json detection object """

        if class_label_map is not None:
            self.class_label = class_label_map[json_obj['category_id'] - 1]
        else:
            self.class_label = str(json_obj['category_id'])

        self.x_top_left = float(json_obj['bbox'][0])
        self.y_top_left = float(json_obj['bbox'][1])
        self.width = float(json_obj['bbox'][2])
        self.height = float(json_obj['bbox'][3])
        self.confidence = json_obj['score']

        self.object_id = 0


class CocoParser(Parser):
    """
    COCO detection format parser to parse the coco detection output of the darknet_ DL framework.

    Keyword Args:
        class_label_map (list): list of class label strings where the ``category_id`` in the json file \
        is used as an index minus one on this list to get the class labels

    A text file contains multiple detections formated using json.
    The file contains one json list where each element represents one bounding box.
    The fields within the elements are:

    ===========  ===========
    Name         Description
    ===========  ===========
    image_id     identifier of the image (integer)
    category_id  class label index (where 1 is the first class label i.s.o. 0) (integer)
    bbox         json list containing bounding box coordinates [top left x, top left y, width, height] (float values)
    score        confidence score between 0 and 1 (float)
    ===========  ===========

    Example:
        >>> detection_results.json
            [
              {"image_id":0, "category_id":1, "bbox":[501.484039, 209.805313, 28.525848, 50.727005], "score":0.189649},
              {"image_id":1, "category_id":1, "bbox":[526.957703, 219.587631, 25.830444, 55.723373], "score":0.477851}
            ]

    .. _darknet: https://pjreddie.com/darknet/
    """
    parser_type = ParserType.SINGLE_FILE
    box_type = CocoDetection
    extension = '.json'

    def __init__(self, **kwargs):
        try:
            self.class_label_map = kwargs['class_label_map']
        except KeyError:
            raise ValueError("Coco detection format requires a 'class_label_map' kwarg")

    def serialize(self, detections):
        """ Serialize input detection to a json string """

        raise NotImplementedError

    def deserialize(self, string):
        """ Parse a json string into a dictionary of detections """
        json_obj = json.loads(string)

        result = {}
        for json_det in json_obj:
            img_id = json_det['image_id']
            if img_id not in result:
                result[img_id] = []
            det = self.box_type()
            det.deserialize(json_det, self.class_label_map)
            result[img_id] += [det]

        return result
