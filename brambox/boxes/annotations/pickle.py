#
#   Copyright EAVISE
#   Author: Tanguy Ophoff
#
"""
Pickle
------
"""
import logging as log
import pickle
from .annotation import *

__all__ = ['PickleAnnotation', 'PickleParser']
#log = logging.getLogger(__name__)


class PickleAnnotation(Annotation):
    """ Pickle annotation """
    def __getstate__(self):
        state = self.__dict__.copy()
        if hasattr(self, 'keep_ignore') and not self.keep_ignore:
            del state['ignore']
        if hasattr(self, 'keep_ignore'):
            del state['keep_ignore']
        if self.visible_x_top_left == 0:
            del state['visible_x_top_left']
        if self.visible_y_top_left == 0:
            del state['visible_y_top_left']
        if self.visible_width == 0:
            del state['visible_width']
        if self.visible_height == 0:
            del state['visible_height']

        return state

    def __setstate__(self, state):
        if 'occluded_fraction' not in state:   # Backward compatible with older versions
            log.deprecated('You are using an old pickle format that will be deprecated in newer versions. Consider to save your annotations with the new format.')
            state['occluded_fraction'] = float(state['occluded'])
            del state['occluded']
        if 'truncated_fraction' not in state:   # Backward compatible with older versions
            log.deprecated('You are using an old pickle format that will be deprecated in newer versions. Consider to save your annotations with the new format.')
            state['truncated_fraction'] = 0.0

        self.__dict__.update(state)
        if not hasattr(self, 'ignore'):
            self.ignore = False
        if not hasattr(self, 'visible_x_top_left'):
            self.visible_x_top_left = 0.0
        if not hasattr(self, 'visible_y_top_left'):
            self.visible_y_top_left = 0.0
        if not hasattr(self, 'visible_width'):
            self.visible_width = 0.0
        if not hasattr(self, 'visible_height'):
            self.visible_height = 0.0


class PickleParser(Parser):
    """
    This parser generates a binary file of your annotations that can be parsed really fast.
    If you are using a python library for training your network, you can use this format to quickly read your annotations.

    Args:
        keep_ignore (boolean, optional): Whether are not to save the ignore flag value of the annotations; Default **False**
    """
    parser_type = ParserType.SINGLE_FILE
    box_type = PickleAnnotation
    extension = '.pkl'
    read_mode = 'rb'
    write_mode = 'wb'

    def __init__(self, **kwargs):
        try:
            self.keep_ignore = kwargs['keep_ignore']
        except KeyError:
            log.info("No 'keep_ignore' kwarg found, defaulting to False.")
            self.keep_ignore = False

    def serialize(self, annotations):
        """ Serialize input dictionary of annotations into one bytestream """
        result = {}
        for img_id in annotations:
            img_res = []
            for anno in annotations[img_id]:
                box = self.box_type.create(anno)
                box.keep_ignore = self.keep_ignore
                img_res.append(box)
            result[img_id] = img_res

        return pickle.dumps(result)

    def deserialize(self, bytestream):
        """ Deserialize an annotation file into a dictionary of annotations """
        return pickle.loads(bytestream)
