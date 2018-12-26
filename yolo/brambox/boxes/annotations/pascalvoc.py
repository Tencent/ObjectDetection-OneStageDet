#
#   Copyright EAVISE
#   By Tanguy Ophoff
#
"""
Pascal VOC
----------
"""

import xml.etree.ElementTree as ET

from .annotation import *

__all__ = ['PascalVocAnnotation', 'PascalVocParser']


class PascalVocAnnotation(Annotation):
    """ Pascal Voc image annotation """
    def serialize(self):
        """ generate a Pascal Voc object xml string """
        string = '<object>\n'
        string += f'\t<name>{self.class_label}</name>\n'
        string += '\t<pose>Unspecified</pose>\n'
        string += f'\t<truncated>{int(self.occluded)}</truncated>\n'
        string += f'\t<difficult>{int(self.difficult)}</difficult>\n'
        string += '\t<bndbox>\n'
        string += f'\t\t<xmin>{self.x_top_left}</xmin>\n'
        string += f'\t\t<ymin>{self.y_top_left}</ymin>\n'
        string += f'\t\t<xmax>{self.x_top_left + self.width - 1}</xmax>\n'
        string += f'\t\t<ymax>{self.y_top_left + self.height - 1}</ymax>\n'
        string += '\t</bndbox>\n'
        string += '</object>\n'

        return string

    def deserialize(self, xml_obj):
        """ parse a Pascal Voc xml annotation string """
        self.class_label = xml_obj.find('name').text
        self.occluded = xml_obj.find('truncated').text == '1'
        self.difficult = xml_obj.find('difficult').text == '1'

        box = xml_obj.find('bndbox')
        self.x_top_left = float(box.find('xmin').text)
        self.y_top_left = float(box.find('ymin').text)
        self.width = float(int(box.find('xmax').text) - self.x_top_left + 1)
        self.height = float(int(box.find('ymax').text) - self.y_top_left + 1)

        self.object_id = 0
        self.lost = None

        return self


class PascalVocParser(Parser):
    """
    This parser can parse annotations in the `pascal voc`_ format.
    This format consists of one xml file for every image.

    Example:
        >>> image_000.xml
            <annotation>
              <object>
                <name>horse</name>
                <truncated>1</truncated>
                <difficult>0</difficult>
                <bndbox>
                  <xmin>100</xmin>
                  <ymin>200</ymin>
                  <xmax>300</xmax>
                  <ymax>400</ymax>
                </bndbox>
              </object>
              <object>
                <name>person</name>
                <truncated>0</truncated>
                <difficult>1</difficult>
                <bndbox>
                  <xmin>110</xmin>
                  <ymin>20</ymin>
                  <xmax>200</xmax>
                  <ymax>350</ymax>
                </bndbox>
              </object>
            </annotation>

    .. _pascal voc: http://host.robots.ox.ac.uk/pascal/VOC/
    """
    parser_type = ParserType.MULTI_FILE
    box_type = PascalVocAnnotation
    extension = '.xml'

    def serialize(self, annotations):
        """ Serialize a list of annotations into one string """
        result = '<annotation>\n'

        for anno in annotations:
            new_anno = self.box_type.create(anno)
            result += new_anno.serialize()

        return result + '</annotation>\n'

    def deserialize(self, string):
        """ Deserialize an annotation string into a list of annotation """
        result = []

        root = ET.fromstring(string)
        for obj in root.iter('object'):
            anno = self.box_type()
            result += [anno.deserialize(obj)]

        return result
