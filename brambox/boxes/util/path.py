#
#   Copyright EAVISE
#   Author: Maarten Vandersteegen
#

import os
import glob

__all__ = ["expand"]


def files(path):
    """ List all files in a directory omitting directories. """
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield os.path.join(path, file)


def strider(elements, stride, offset):
    """ Yield input elements with given stride and offset. """
    next_element = offset

    # support negative offsets
    while next_element < 0:
        next_element += stride

    for i, elem in enumerate(elements):
        if i == next_element:
            next_element += stride
            yield elem


def modulo_expand(expr, stride, offset):
    """ Expands a path with a **%d** to files with different numbers. """
    # Support negative offset
    number = offset
    while (number < 0):
        number += stride

    while True:
        filename = expr % number
        if not os.path.isfile(filename):
            break
        yield filename
        number += stride


def expand(expr, stride=1, offset=0):
    """ Expand a file selection expression into multiple filenames.

    Args:
        expr (str): File sequence expression
        stride (int, optional): Sample every n'th file where n is this parameter; Default **1**
        offset (int, optional): Start with the m'th file where m is this parameter; Default **0**

    Returns:
        generator: Iterable object that produces full filenames

    Note:
        The ``expr`` parameter can be one of the following expressions:

        - a file itself -> return filename
        - a directory -> return files from directory
        - path with **'*'** wildcard -> return globbed files
        - path with **'%d'** wildcard -> return incremental files
    """
    if os.path.isdir(expr):
        return strider(sorted(files(expr)), stride, offset)
    elif os.path.isfile(expr):
        return [expr]
    elif '*' in expr:
        return strider(sorted(glob.glob(expr)), stride, offset)
    elif '%' in expr:
        return modulo_expand(expr, stride, offset)
    else:
        raise TypeError("File selection expression invalid")
