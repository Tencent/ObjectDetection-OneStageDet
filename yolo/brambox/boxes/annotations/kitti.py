#
#   Copyright EAVISE
#   Author: Tanguy Ophoff
#
"""
KITTI
-----
"""

from .annotation import *

__all__ = ["KittiAnnotation", "KittiParser"]


class KittiAnnotation(Annotation):
    """ KITI image annotation """
    def serialize(self):
        """ generate a KITTI annotation string """
        truncated = 1.0 if self.lost else self.truncated_fraction
        class_label = self.class_label if self.class_label != '' else '?'
        if self.occluded_fraction >= 0.5:
            occluded = 2
        elif self.occluded_fraction > 0.0:
            occluded = 1
        else:
            occluded = 0

        return f'{class_label} {truncated:.2f} {occluded} -10 {self.x_top_left:.2f} {self.y_top_left:.2f} {self.x_top_left+self.width:.2f} {self.y_top_left+self.height:.2f} -1 -1 -1 -1000 -1000 -1000 -10'

    def deserialize(self, string):
        """ parse a KITTI annotation string """
        elements = string.split()

        self.class_label = elements[0] if elements[0] != '?' else ''
        self.truncated_fraction = max(float(elements[1]), 0.0)
        self.x_top_left = float(elements[4])
        self.y_top_left = float(elements[5])
        self.width = float(elements[6]) - self.x_top_left
        self.height = float(elements[7]) - self.y_top_left

        if elements[2] == '1':
            self.occluded_fraction = 0.25
        elif elements[2] == '2':
            self.occluded_fraction = 0.5
        else:
            self.occluded_fraction = 0.0


class KittiParser(Parser):
    """
    This parser can read and write kitti_ annotation files. |br|
    Some of the values of this dataset are not present in the brambox annotation objects and are thus not used.
    When serializing this format, these values will be set to their default value, as per specification.

    ==================  ================  ===========
    Name                Number of Values  Description
    ==================  ================  ===========
    class_label         1                 Annotation class_label. In the official dataset this can be one of: |br|
                                          'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare'

    truncated_fraction  1                 Float in range [0-1] indicating whether object is truncated

    occluded_state      1                 Integer (0,1,2,3) indicating occlusion state: |br|
                                          0=fully visible, 1=partly occluded, 2=largely occluded, 3=unknown

    alpha               1                 *[Not used in brambox]* Observation angle of the object

    bbox                4                 2D bounding box of the image, expressed in pixel coordinates

    dimensions          3                 *[Not used in brambox]* 3D object dimensions

    location            3                 *[Not used in brambox]* 3D object location

    rotation_y          1                 *[Not used in brambox]* Rotation around Y-axis in camera coordinates
    ==================  ================  ===========

    Example:
        >>> image_000.txt
            <class_label> <truncated_fraction> <occluded_state> -10 <bbox_left> <bbox_top> <bbox_right> <bbox_bottom> -1 -1 -1 -1000 -1000 -1000 -10
            <class_label> <truncated_fraction> <occluded_state> -10 <bbox_left> <bbox_top> <bbox_right> <bbox_bottom> -1 -1 -1 -1000 -1000 -1000 -10
        >>> image_001.txt
            <class_label> <truncated_fraction> <occluded_state> -10 <bbox_left> <bbox_top> <bbox_right> <bbox_bottom> -1 -1 -1 -1000 -1000 -1000 -10
            <class_label> <truncated_fraction> <occluded_state> -10 <bbox_left> <bbox_top> <bbox_right> <bbox_bottom> -1 -1 -1 -1000 -1000 -1000 -10
            <class_label> <truncated_fraction> <occluded_state> -10 <bbox_left> <bbox_top> <bbox_right> <bbox_bottom> -1 -1 -1 -1000 -1000 -1000 -10

    Note:
        This parser will convert the ``occluded_state`` to an ``occluded_fraction``. |br|
        Partly occluded (1) will be converted to a fraction of 0.25 and largely occluded (2) to 0.5.
        The other states will be converted to a fraction of 0. |br|
        When serializing, all fractions bigger or equal to 0.5 will be converted to largely occluded (2),
        fractions between 0.5 and 0 to partly occluded (1) and fractions of 0 will be converted to fully visible (0).

    .. _kitti: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
    """
    parser_type = ParserType.MULTI_FILE
    box_type = KittiAnnotation
