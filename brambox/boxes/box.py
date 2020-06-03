#
#   Copyright EAVISE
#

from enum import Enum

__all__ = ['Box', 'ParserType', 'Parser']


class Box:
    """ This is a generic bounding box representation.
    This class provides some base functionality to both annotations and detections.

    Attributes:
        class_label (string): class string label; Default **''**
        object_id (int): Object identifier for reid purposes; Default **0**
        x_top_left (Number): X pixel coordinate of the top left corner of the bounding box; Default **0.0**
        y_top_left (Number): Y pixel coordinate of the top left corner of the bounding box; Default **0.0**
        width (Number): Width of the bounding box in pixels; Default **0.0**
        height (Number): Height of the bounding box in pixels; Default **0.0**
    """
    def __init__(self):
        self.class_label = ''   # class string label
        self.object_id = 0      # object identifier
        self.x_top_left = 0.0   # x pixel coordinate top left of the box
        self.y_top_left = 0.0   # y pixel coordinate top left of the box
        self.width = 0.0        # width of the box in pixels
        self.height = 0.0       # height of the box in pixels

    @classmethod
    def create(cls, obj=None):
        """ Create a bounding box from a string or other detection object.

        Args:
            obj (Box or string, optional): Bounding box object to copy attributes from or string to deserialize
        """
        instance = cls()

        if obj is None:
            return instance

        if isinstance(obj, str):
            instance.deserialize(obj)
        elif isinstance(obj, Box):
            instance.class_label = obj.class_label
            instance.object_id = obj.object_id
            instance.x_top_left = obj.x_top_left
            instance.y_top_left = obj.y_top_left
            instance.width = obj.width
            instance.height = obj.height
        else:
            raise TypeError(f'Object is not of type Box or not a string [obj.__class__.__name__]')

        return instance

    def __eq__(self, other):
        # TODO: refactor -> use almost equal for floats
        return self.__dict__ == other.__dict__

    def serialize(self):
        """ abstract serializer, implement in derived classes. """
        raise NotImplementedError

    def deserialize(self, string):
        """ abstract parser, implement in derived classes. """
        raise NotImplementedError


class ParserType(Enum):
    """ Enum for differentiating between different parser types. """
    UNDEFINED = 0       #: Undefined parsertype. Do not use this!
    SINGLE_FILE = 1     #: One single file contains all annotations
    MULTI_FILE = 2      #: One annotation file per image


class Parser:
    """ This is a Generic parser class.

    Args:
        kwargs (optional): Derived parsers should use keyword arguments to get any information they need upon initialisation.
    """
    parser_type = ParserType.UNDEFINED  #: Type of parser. Derived classes should set the correct value.
    box_type = Box                      #: Type of bounding box this parser parses or generates. Derived classes should set the correct type.
    extension = '.txt'                  #: Extension of the files this parser parses or creates. Derived classes should set the correct extension.
    read_mode = 'r'                     #: Reading mode this parser uses when it parses a file. Derived classes should set the correct mode.
    write_mode = 'w'                    #: Writing mode this parser uses when it generates a file. Derived classes should set the correct mode.

    def __init__(self, **kwargs):
        pass

    def serialize(self, box):
        """ Serialization function that can be overloaded in the derived class.
        The default serializer will call the serialize function of the bounding boxes and join them with a newline.

        Args:
            box: Bounding box objects

        Returns:
            string: Serialized bounding boxes

        Note:
            The format of the box parameter depends on the type of parser. |br|
            If it is a :any:`brambox.boxes.box.ParserType.SINGLE_FILE`, the box parameter should be a dictionary ``{"image_id": [box, box, ...], ...}``. |br|
            If it is a :any:`brambox.boxes.box.ParserType.MULTI_FILE`, the box parameter should be a list ``[box, box, ...]``.
        """
        if self.parser_type != ParserType.MULTI_FILE:
            raise TypeError('The default implementation of serialize only works with MULTI_FILE')

        result = ""
        for b in box:
            new_box = self.box_type.create(b)
            result += new_box.serialize() + "\n"

        return result

    def deserialize(self, string):
        """ Deserialization function that can be overloaded in the derived class.
        The default deserialize will create new ``box_type`` objects and call the deserialize function of these objects with every line of the input string.

        Args:
            string (string): Input string to deserialize

        Returns:
            box: Bounding box objects

        Note:
            The format of the box return value depends on the type of parser. |br|
            If it is a :any:`brambox.boxes.box.ParserType.SINGLE_FILE`, the return value should be a dictionary ``{"image_id": [box, box, ...], ...}``. |br|
            If it is a :any:`brambox.boxes.box.ParserType.MULTI_FILE`, the return value should be a list ``[box, box, ...]``.
        """
        if self.parser_type != ParserType.MULTI_FILE:
            raise TypeError('The default implementation of deserialize only works with MULTI_FILE')

        result = []
        for line in string.splitlines():
            result += [self.box_type.create(line)]

        return result
