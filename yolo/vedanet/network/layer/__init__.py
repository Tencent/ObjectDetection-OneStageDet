#
#   Extra lightnet layers
#   Copyright EAVISE
#
"""
.. Note::
   Every parameter that can get an int or tuple will behave as follows. |br|
   If a tuple of 2 ints is given, the first int is used for the height and the second for the width. |br|
   If an int is given, both the width and height are set to this value.
"""

from ._darknet import *
from ._mobilenet import *
