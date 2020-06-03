#
#   Copyright EAVISE
#

from .annotations import annotation_formats
from .detections import detection_formats

__all__ = ['formats', 'annotation_formats', 'detection_formats']

formats = {}
for key in annotation_formats:
    formats['anno_'+key] = annotation_formats[key]
for key in detection_formats:
    formats['det_'+key] = detection_formats[key]
