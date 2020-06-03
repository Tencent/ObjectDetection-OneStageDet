"""
Lightnet Engine Module |br|
This module contains classes and functions to manage the training of your networks.
It has an engine, capable of orchestrating your training and test cycles, and also contains function to easily visualise data with visdom_.
"""


#from .engine import *
from ._voc_train import *
from ._voc_test import *
from ._speed import *
