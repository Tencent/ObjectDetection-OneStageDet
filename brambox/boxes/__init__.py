"""
Brambox boxes module |br|
This package contains parsers for various annotation and detection formats.
You can use this package to convert formats, visualize image annotations and compute statistics on your detections.
"""

from .box import Box
from .formats import *
from . import annotations
from . import detections

from .statistics import *
from .util import *
