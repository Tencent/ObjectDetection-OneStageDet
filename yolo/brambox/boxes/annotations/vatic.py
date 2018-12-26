#
#   Copyright EAVISE
#   Author: Maarten Vandersteegen
#
"""
Vatic
-----
"""

from .annotation import *

__all__ = ["VaticAnnotation", "VaticParser"]


class VaticAnnotation(Annotation):
    """ VATIC tool annotation """

    def serialize(self, frame_nr=0):
        """ generate a vatic annotation string """

        object_id = self.object_id
        x_min = round(self.x_top_left)
        y_min = round(self.y_top_left)
        x_max = round(self.x_top_left + self.width)
        y_max = round(self.y_top_left + self.height)
        lost = int(self.lost)
        occluded = int(self.occluded)
        generated = 0
        class_label = '?' if self.class_label == '' else self.class_label

        string = "{} {} {} {} {} {} {} {} {} {}" \
            .format(object_id,
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    frame_nr,
                    lost,
                    occluded,
                    generated,
                    class_label)

        return string

    def deserialize(self, string):
        """ parse a valitc annotation """

        elements = string.split()
        self.object_id = int(elements[0])
        self.x_top_left = float(elements[1])
        self.y_top_left = float(elements[2])
        self.width = abs(float(elements[3]) - self.x_top_left)
        self.height = abs(float(elements[4]) - self.y_top_left)
        frame_nr = int(elements[5])
        self.lost = elements[6] != '0'
        self.occluded = elements[7] != '0'
        self.class_label = elements[9].strip('\"')
        if self.class_label == '?':
            self.class_label = ''


class VaticParser(Parser):
    """
    This parser is designed to parse the standard VATIC_ video annotation tool text files.
    The VATIC format contains all annotation from multiple images into one file.
    Each line of the file represents one bounding box from one image and is a spaces separated
    list of values structured as follows:

        <track_id> <xmin> <ymin> <xmax> <ymax> <frame> <lost> <occluded> <generated> <label>

    =========  ===========
    Name       Description
    =========  ===========
    track_id   identifier of the track this object is following (integer)
    xmin       top left x coordinate of the bounding box (integer)
    ymin       top left y coordinate of the bounding box (integer)
    xmax       bottom right x coordinate of the bounding box (integer)
    ymax       bottom right y coordinate of the bounding box (integer)
    frame      image identifier that this annotation belong to (integer)
    lost       1 if the annotated object is outside of the view screen, 0 otherwise
    occluded   1 if the annotated object is occluded, 0 otherwise
    generated  1 if the annotation was automatically interpolated, 0 otherwise (not used)
    label      class label of the object, enclosed in quotation marks
    =========  ===========

    Example:
        >>> video_000.txt
            1 578 206 762 600 282 0 0 0 "person"
            2 206 286 234 340 0 1 0 0 "person"
            8 206 286 234 340 10 1 0 1 "car"

    .. _VATIC: https://github.com/cvondrick/vatic
    """
    parser_type = ParserType.SINGLE_FILE
    box_type = VaticAnnotation

    def serialize(self, annotations):
        """ Serialize input dictionary of annotations into a VATIC annotation string """

        result = []
        for img_id, annos in annotations.items():
            for anno in annos:
                new_anno = self.box_type.create(anno)
                result += [new_anno.serialize(img_id)]

        return "\n".join(result)

    def deserialize(self, string):
        """ deserialize a string containing the content of a VATIC .txt file """

        result = {}
        for line in string.splitlines():
            img_id = line.split()[5]
            if img_id not in result:
                result[img_id] = []

            anno = self.box_type()
            anno.deserialize(line)
            result[img_id] += [anno]

        return result
