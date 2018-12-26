#
#   Copyright EAVISE
#   Author: Maarten Vandersteegen
#   Author: Tanguy Ophoff
#

import os
from .path import expand
from ..formats import formats
from ..box import ParserType, Parser, Box

__all__ = ['parse', 'generate']


def parse(fmt, box_file, identify=None, offset=0, stride=1, **kwargs):
    """ Parse any type of bounding box format.

    Args:
        fmt (str or class): Format from the :mod:`brambox.boxes.format <brambox.boxes>` dictionary
        box_file (list or string): Bounding box filename or array of bounding box file names
        identify (function, optional): Function to create an image identifier
        offset (int, optional): Skip images untill offset; Default **0**
        stride (int, optional): Only read every n'th file; Default **1**
        **kwargs: Keyword arguments that are passed to the parser

    Returns:
        dict: Dictionary containing the bounding boxes for every image ``{"image_id": [box, box, ...], ...}``

    Note:
        The ``identify`` function will be used to generate ``image_id`` tags. |br|
        If the format is of the type :any:`brambox.boxes.box.ParserType.SINGLE_FILE`,
        the identify function gets the existing ``image_id`` tags as input. The default is to not change the tags. |br|
        If the format is of the type :any:`brambox.boxes.box.ParserType.MULTI_FILE`,
        the identify function gets the path of the current file as input. The default is to get the name of the file without extensions.

    Warning:
        The ``box_file`` parameter can be either a list or string. |br|
        If the format is of the type :any:`brambox.boxes.box.ParserType.SINGLE_FILE`,
        then only a string is accepted and this is used as the filename. |br|
        If the format is of the type :any:`brambox.boxes.box.ParserType.MULTI_FILE`,
        then you can either pass a list or a string.
        A list will be used as is, namely every string of the list gets used as a filename.
        If you use a string, it will first be expanded with the :func:`~brambox.boxes.expand` function
        to generate a list of strings. This expand function can take optional stride and offset parameters,
        which can be passed via keyword arguments.
    """

    # Create parser
    if type(fmt) is str:
        try:
            parser = formats[fmt](**kwargs)
        except KeyError:
            raise TypeError(f'Invalid parser {fmt}')
    elif issubclass(fmt, Parser):
        parser = fmt(**kwargs)
    else:
        raise TypeError(f'Invalid parser {fmt}')

    # Parse bounding boxes
    if parser.parser_type == ParserType.SINGLE_FILE:
        if type(box_file) is not str:
            raise TypeError(f'Parser <{parser.__class__.__name__}> requires a single annotation file')
        with open(box_file, parser.read_mode) as f:
            data = parser.deserialize(f.read())

        # Offset
        if offset > 0:
            keys = sorted(list(data.keys()))
            while offset > 0:
                offset -= 1
                del data[keys[offset]]

        # Stride
        if stride > 1:
            new_data = {}
            keys = sorted(list(data.keys()))
            length = len(keys)
            number = offset

            while number < 0:
                number += stride

            while number < length:
                new_data[keys[number]] = data[keys[number]]
                number += stride

            data = new_data

        # Identify
        if identify is not None:
            data = {identify(key): value for key, value in data.items()}
    elif parser.parser_type == ParserType.MULTI_FILE:
        if type(box_file) is str:
            box_files = expand(box_file, stride, offset)
        elif type(box_file) is list:
            box_files = box_file
        else:
            raise TypeError(f'Parser <{parser.__class__.__name__}> requires a list of annotation files or an expandable file expression')

        # Default identify
        if identify is None:
            def identify(f): return os.path.splitext(os.path.basename(f))[0]

        data = {}
        for box_file in box_files:
            img_id = identify(box_file)
            if img_id in data:
                raise ValueError(f'Multiple bounding box files with the same name were found ({img_id})')

            with open(box_file, parser.read_mode) as f:
                data[img_id] = parser.deserialize(f.read())
    else:
        raise AttributeError(f'Parser <{parser.__class__.__name__}> has not defined a parser_type class attribute')

    return data


def generate(fmt, box, path, **kwargs):
    """ Generate bounding box file(s) in any format.

    Args:
        fmt (str or class): Format from the :mod:`brambox.boxes.format <brambox.boxes>` dictionary
        box (dict): Dictionary containing box objects per image ``{"image_id": [box, box, ...], ...}``
        path (str): Path to the bounding box file/folder
        **kwargs (dict): Keyword arguments that are passed to the parser

    Warning:
        If the format is of the type :any:`brambox.boxes.box.ParserType.SINGLE_FILE`,
        then the ``path`` parameter should contain a path to a **file**. |br|
        If the format is of the type :any:`brambox.boxes.box.ParserType.MULTI_FILE`,
        then the ``path`` parameter should contain a path to a **folder**.
    """

    # Create parser
    if type(fmt) is str:
        try:
            parser = formats[fmt](**kwargs)
        except KeyError:
            raise TypeError(f'Invalid parser {fmt}')
    elif issubclass(fmt, Parser):
        parser = fmt(**kwargs)
    else:
        raise TypeError(f'Invalid parser {fmt}')

    # Write bounding boxes
    if parser.parser_type == ParserType.SINGLE_FILE:
        if os.path.isdir(path):
            path = os.path.join(path, 'boxes' + parser.extension)
        with open(path, parser.write_mode) as f:
            f.write(parser.serialize(box))
    elif parser.parser_type == ParserType.MULTI_FILE:
        if not os.path.isdir(path):
            raise ValueError(f'Parser <{parser.__class__.__name__}> requires a path to a folder')
        for img_id, boxes in box.items():
            filename = os.path.join(path, img_id + parser.extension)

            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                os.makedirs(directory)

            with open(filename, parser.write_mode) as f:
                f.write(parser.serialize(boxes))
    else:
        raise AttributeError(f'Parser <{parser.__class__.__name__}> has not defined a parser_type class attribute')
