#
#   Copyright EAVISE
#   Author: Tanguy Ophoff
#
"""
Pickle
------
"""

import pickle
from .detection import *

__all__ = ["PickleParser"]


class PickleParser(Parser):
    """
    This parser generates a binary file of your detections that can be parsed really fast.
    If you are using a python library for testing your network, you can use this format to quickly save your detections.
    """
    parser_type = ParserType.SINGLE_FILE
    box_type = Detection
    extension = '.pkl'
    read_mode = 'rb'
    write_mode = 'wb'

    def serialize(self, annotations):
        """ Serialize input dictionary of annotations into one bytestream """
        result = {}
        for img_id in annotations:
            img_res = []
            for anno in annotations[img_id]:
                img_res.append(self.box_type.create(anno))
            result[img_id] = img_res

        return pickle.dumps(result)

    def deserialize(self, bytestream):
        """ Deserialize an annotation file into a dictionary of annotations """
        return pickle.loads(bytestream)
