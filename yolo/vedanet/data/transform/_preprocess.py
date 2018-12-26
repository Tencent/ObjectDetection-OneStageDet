#
#   Image and annotations preprocessing for lightnet networks
#   The image transformations work with both Pillow and OpenCV images
#   The annotation transformations work with brambox.annotations.Annotation objects
#   Copyright EAVISE
#

# modified by mileistone

import random
import collections
import logging as log
import torch
import numpy as np
from PIL import Image, ImageOps
import brambox.boxes as bbb
from .util import BaseTransform, BaseMultiTransform

try:
    import cv2
except ImportError:
    log.warn('OpenCV is not installed and cannot be used')
    cv2 = None

__all__ = ['Letterbox', 'RandomCrop', 'RandomCropLetterbox', 'RandomFlip', 'HSVShift', 'BramboxToTensor']


class Letterbox(BaseMultiTransform):
    """ Transform images and annotations to the right network dimensions.

    Args:
        dimension (tuple, optional): Default size for the letterboxing, expressed as a (width, height) tuple; Default **None**
        dataset (lightnet.data.Dataset, optional): Dataset that uses this transform; Default **None**

    Note:
        Create 1 Letterbox object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, dimension=None, dataset=None):
        super().__init__(dimension=dimension, dataset=dataset)
        if self.dimension is None and self.dataset is None:
            raise ValueError('This transform either requires a dimension or a dataset to infer the dimension')

        self.pad = None
        self.scale = None
        self.fill_color = 127

    def __call__(self, data):
        if data is None:
            return None
        elif isinstance(data, collections.Sequence):
            return self._tf_anno(data)
        elif isinstance(data, Image.Image):
            return self._tf_pil(data)
        elif isinstance(data, np.ndarray):
            return self._tf_cv(data)
        else:
            log.error(f'Letterbox only works with <brambox annotation lists>, <PIL images> or <OpenCV images> [{type(data)}]')
            return data

    def _tf_pil(self, img):
        """ Letterbox an image to fit in the network """
        if self.dataset is not None:
            net_w, net_h = self.dataset.input_dim
        else:
            net_w, net_h = self.dimension
        im_w, im_h = img.size

        if im_w == net_w and im_h == net_h:
            self.scale = None
            self.pad = None
            return img

        # Rescaling
        if im_w / net_w >= im_h / net_h:
            self.scale = net_w / im_w
        else:
            self.scale = net_h / im_h
        if self.scale != 1:
            resample_mode = Image.NEAREST #Image.BILINEAR if self.scale > 1 else Image.ANTIALIAS
            img = img.resize((int(self.scale*im_w), int(self.scale*im_h)), resample_mode)
            im_w, im_h = img.size

        if im_w == net_w and im_h == net_h:
            self.pad = None
            return img

        # Padding
        img_np = np.array(img)
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
        pad_w = (net_w - im_w) / 2
        pad_h = (net_h - im_h) / 2
        self.pad = (int(pad_w), int(pad_h), int(pad_w+.5), int(pad_h+.5))
        img = ImageOps.expand(img, border=self.pad, fill=(self.fill_color,)*channels)
        return img

    def _tf_cv(self, img):
        """ Letterbox and image to fit in the network """
        if self.dataset is not None:
            net_w, net_h = self.dataset.input_dim
        else:
            net_w, net_h = self.dimension
        im_h, im_w = img.shape[:2]

        if im_w == net_w and im_h == net_h:
            self.scale = None
            self.pad = None
            return img

        # Rescaling
        if im_w / net_w >= im_h / net_h:
            self.scale = net_w / im_w
        else:
            self.scale = net_h / im_h
        if self.scale != 1:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
            im_h, im_w = img.shape[:2]

        if im_w == net_w and im_h == net_h:
            self.pad = None
            return img

        # Padding
        channels = img.shape[2] if len(img.shape) > 2 else 1
        pad_w = (net_w - im_w) / 2
        pad_h = (net_h - im_h) / 2
        self.pad = (int(pad_w), int(pad_h), int(pad_w+.5), int(pad_h+.5))
        img = cv2.copyMakeBorder(img, self.pad[1], self.pad[3], self.pad[0], self.pad[2], cv2.BORDER_CONSTANT, value=(self.fill_color,)*channels)
        return img

    def _tf_anno(self, annos):
        """ Change coordinates of an annotation, according to the previous letterboxing """
        for anno in annos:
            if self.scale is not None:
                anno.x_top_left *= self.scale
                anno.y_top_left *= self.scale
                anno.width *= self.scale
                anno.height *= self.scale
            if self.pad is not None:
                anno.x_top_left += self.pad[0]
                anno.y_top_left += self.pad[1]
        return annos


class RandomCrop(BaseMultiTransform):
    """ Take random crop from the image.

    Args:
        jitter (Number [0-1]): Indicates how much of the image we can crop
        crop_anno(Boolean, optional): Whether we crop the annotations inside the image crop; Default **False**
        intersection_threshold(number or list, optional): Argument passed on to :class:`brambox.boxes.util.modifiers.CropModifier`

    Note:
        Create 1 RandomCrop object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, jitter, crop_anno=False, intersection_threshold=0.001, fill_color=127):
        super().__init__(jitter=jitter, crop_anno=crop_anno, fill_color=fill_color)
        self.crop_modifier = bbb.CropModifier(float('Inf'), intersection_threshold)

    def __call__(self, data):
        if data is None:
            return None
        elif isinstance(data, collections.Sequence):
            return self._tf_anno(data)
        elif isinstance(data, Image.Image):
            return self._tf_pil(data)
        elif isinstance(data, np.ndarray):
            return self._tf_cv(data)
        else:
            log.error(f'RandomCrop only works with <brambox annotation lists>, <PIL images> or <OpenCV images> [{type(data)}]')
            return data

    def _tf_pil(self, img):
        """ Take random crop from image """
        im_w, im_h = img.size
        crop = self._get_crop(im_w, im_h)
        crop_w = crop[2] - crop[0]
        crop_h = crop[3] - crop[1]
        img_np = np.array(img)
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1

        img = img.crop((max(0, crop[0]), max(0, crop[1]), min(im_w, crop[2]-1), min(im_h, crop[3]-1)))
        img_crop = Image.new(img.mode, (crop_w, crop_h), color=(self.fill_color,)*channels)
        img_crop.paste(img, (max(0, -crop[0]), max(0, -crop[1])))

        return img_crop

    def _tf_cv(self, img):
        """ Take random crop from image """
        im_h, im_w = img.shape[:2]
        crop = self._get_crop(im_w, im_h)

        crop_w = crop[2] - crop[0]
        crop_h = crop[3] - crop[1]
        img_crop = np.ones((crop_h, crop_w) + img.shape[2:], dtype=img.dtype) * self.fill_color

        src_x1 = max(0, crop[0])
        src_x2 = min(crop[2], im_w)
        src_y1 = max(0, crop[1])
        src_y2 = min(crop[3], im_h)
        dst_x1 = max(0, -crop[0])
        dst_x2 = crop_w - max(0, crop[2]-im_w)
        dst_y1 = max(0, -crop[1])
        dst_y2 = crop_h - max(0, crop[3]-im_h)
        img_crop[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

        return img_crop

    def _get_crop(self, im_w, im_h):
        dw, dh = int(im_w*self.jitter), int(im_h*self.jitter)
        crop_left = random.randint(-dw, dw)
        crop_right = random.randint(-dw, dw)
        crop_top = random.randint(-dh, dh)
        crop_bottom = random.randint(-dh, dh)
        crop = (crop_left, crop_top, im_w-crop_right, im_h-crop_bottom)

        self.crop_modifier.area = crop
        return crop

    def _tf_anno(self, annos):
        """ Change coordinates of an annotation, according to the previous crop """
        if self.crop_anno:
            bbb.modify(annos, [self.crop_modifier])
        else:
            crop = self.crop_modifier.area
            for i in range(len(annos)-1, -1, -1):
                anno = annos[i]
                x1 = max(crop[0], anno.x_top_left)
                x2 = min(crop[2], anno.x_top_left+anno.width)
                y1 = max(crop[1], anno.y_top_left)
                y2 = min(crop[3], anno.y_top_left+anno.height)
                w = x2-x1
                h = y2-y1

                if self.crop_modifier.inter_area:
                    ratio = ((w * h) / (anno.width * anno.height)) < self.crop_modifier.inter_thresh
                else:
                    ratio = (w / anno.width) < self.crop_modifier.inter_thresh[0] or (h / anno.height) < self.crop_modifier.inter_thresh[1]
                if w <= 0 or h <= 0 or ratio:
                    del annos[i]
                    continue

                annos[i].x_top_left -= crop[0]
                annos[i].y_top_left -= crop[1]

        return annos


class RandomCropLetterbox(BaseMultiTransform):
    """ Take random crop from the image.

    Args:
        jitter (Number [0-1]): Indicates how much of the image we can crop
        crop_anno(Boolean, optional): Whether we crop the annotations inside the image crop; Default **False**
        intersection_threshold(number or list, optional): Argument passed on to :class:`brambox.boxes.util.modifiers.CropModifier`

    Note:
        Create 1 RandomCrop object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, dataset, jitter, fill_color=127):
        super().__init__(dataset=dataset, jitter=jitter, fill_color=fill_color)
        self.crop_info = None
        self.output_w = None
        self.output_h = None

    def __call__(self, data):
        if data is None:
            return None
        elif isinstance(data, collections.Sequence):
            return self._tf_anno(data)
        elif isinstance(data, Image.Image):
            return self._tf_pil(data)
        else:
            log.error(f'RandomCrop only works with <brambox annotation lists>, <PIL images> or <OpenCV images> [{type(data)}]')
            return data

    def _tf_pil(self, img):
        """ Take random crop from image """
        self.output_w, self.output_h = self.dataset.input_dim
        #print('output shape: %d, %d' % (self.output_w, self.output_h))
        orig_w, orig_h = img.size
        img_np = np.array(img)
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
        dw = int(self.jitter * orig_w)
        dh = int(self.jitter * orig_h)
        new_ar = float(orig_w + random.randint(-dw, dw)) / (orig_h + random.randint(-dh, dh))
        scale = random.random()*(2-0.25) + 0.25
        if new_ar < 1:
            nh = int(scale * orig_h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * orig_w)
            nh = int(nw / new_ar)

        if self.output_w > nw:
            dx = random.randint(0, self.output_w - nw)
        else:
            dx = random.randint(self.output_w - nw, 0)

        if self.output_h > nh:
            dy = random.randint(0, self.output_h - nh)
        else:
            dy = random.randint(self.output_h - nh, 0)

        nxmin = max(0, -dx)
        nymin = max(0, -dy)
        nxmax = min(nw, -dx + self.output_w - 1)
        nymax = min(nh, -dy + self.output_h - 1)
        sx, sy = float(orig_w)/nw, float(orig_h)/nh
        orig_xmin = int(nxmin * sx)
        orig_ymin = int(nymin * sy)
        orig_xmax = int(nxmax * sx)
        orig_ymax = int(nymax * sy)
        orig_crop = img.crop((orig_xmin, orig_ymin, orig_xmax, orig_ymax))
        orig_crop_resize = orig_crop.resize((nxmax - nxmin, nymax - nymin))
        output_img = Image.new(img.mode, (self.output_w, self.output_h), color=(self.fill_color,)*channels)
        output_img.paste(orig_crop_resize, (0, 0))
        self.crop_info = [sx, sy, nxmin, nymin, nxmax, nymax]
        return output_img

    def _tf_anno(self, annos):
        """ Change coordinates of an annotation, according to the previous crop """
        sx, sy, crop_xmin, crop_ymin, crop_xmax, crop_ymax = self.crop_info
        for i in range(len(annos)-1, -1, -1):
            anno = annos[i]
            x1 = max(crop_xmin, int(anno.x_top_left/sx))
            x2 = min(crop_xmax, int((anno.x_top_left+anno.width)/sx))
            y1 = max(crop_ymin, int(anno.y_top_left/sy))
            y2 = min(crop_ymax, int((anno.y_top_left+anno.height)/sy))
            w = x2-x1
            h = y2-y1

            if w <= 2 or h <= 2: # or w*h/(anno.width*anno.height/sx/sy) <= 0.5:
                del annos[i]
                continue

            annos[i].x_top_left = x1 - crop_xmin
            annos[i].y_top_left = y1 -crop_ymin
            annos[i].width = w
            annos[i].height = h
        return annos


class RandomFlip(BaseMultiTransform):
    """ Randomly flip image.

    Args:
        threshold (Number [0-1]): Chance of flipping the image

    Note:
        Create 1 RandomFlip object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, threshold):
        self.threshold = threshold
        self.flip = False
        self.im_w = None

    def __call__(self, data):
        if data is None:
            return None
        elif isinstance(data, collections.Sequence):
            return [self._tf_anno(anno) for anno in data]
        elif isinstance(data, Image.Image):
            return self._tf_pil(data)
        elif isinstance(data, np.ndarray):
            return self._tf_cv(data)
        else:
            log.error(f'RandomFlip only works with <brambox annotation lists>, <PIL images> or <OpenCV images> [{type(data)}]')
            return data

    def _tf_pil(self, img):
        """ Randomly flip image """
        self._get_flip()
        self.im_w = img.size[0]
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def _tf_cv(self, img):
        """ Randomly flip image """
        self._get_flip()
        self.im_w = img.shape[1]
        if self.flip:
            img = cv2.flip(img, 1)
        return img

    def _get_flip(self):
        self.flip = random.random() < self.threshold

    def _tf_anno(self, anno):
        """ Change coordinates of an annotation, according to the previous flip """
        if self.flip and self.im_w is not None:
            anno.x_top_left = self.im_w - anno.x_top_left - anno.width
        return anno


class HSVShift(BaseTransform):
    """ Perform random HSV shift on the RGB data.

    Args:
        hue (Number): Random number between -hue,hue is used to shift the hue
        saturation (Number): Random number between 1,saturation is used to shift the saturation; 50% chance to get 1/dSaturation in stead of dSaturation
        value (Number): Random number between 1,value is used to shift the value; 50% chance to get 1/dValue in stead of dValue

    Warning:
        If you use OpenCV as your image processing library, make sure the image is RGB before using this transform.
        By default OpenCV uses BGR, so you must use `cvtColor`_ function to transform it to RGB.

    .. _cvtColor: https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#ga397ae87e1288a81d2363b61574eb8cab
    """
    def __init__(self, hue, saturation, value):
        super().__init__(hue=hue, saturation=saturation, value=value)

    @classmethod
    def apply(cls, data, hue, saturation, value):
        dh = random.uniform(-hue, hue)
        ds = random.uniform(1, saturation)
        if random.random() < 0.5:
            ds = 1/ds
        dv = random.uniform(1, value)
        if random.random() < 0.5:
            dv = 1/dv

        if data is None:
            return None
        elif isinstance(data, Image.Image):
            return cls._tf_pil(data, dh, ds, dv)
        elif isinstance(data, np.ndarray):
            return cls._tf_cv(data, dh, ds, dv)
        else:
            log.error(f'HSVShift only works with <PIL images> or <OpenCV images> [{type(data)}]')
            return data

    @staticmethod
    def _tf_pil(img, dh, ds, dv):
        """ Random hsv shift """
        img = img.convert('HSV')
        channels = list(img.split())

        def change_hue(x):
            x += int(dh * 255)
            if x > 255:
                x -= 255
            elif x < 0:
                x += 0
            return x

        channels[0] = channels[0].point(change_hue)
        channels[1] = channels[1].point(lambda i: min(255, max(0, int(i*ds))))
        channels[2] = channels[2].point(lambda i: min(255, max(0, int(i*dv))))

        img = Image.merge(img.mode, tuple(channels))
        img = img.convert('RGB')
        return img

    @staticmethod
    def _tf_cv(img, dh, ds, dv):
        """ Random hsv shift """
        img = img.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        def wrap_hue(x):
            x[x >= 360.0] -= 360.0
            x[x < 0.0] += 360.0
            return x

        img[:, :, 0] = wrap_hue(hsv[:, :, 0] + (360.0 * dh))
        img[:, :, 1] = np.clip(ds * img[:, :, 1], 0.0, 1.0)
        img[:, :, 2] = np.clip(dv * img[:, :, 2], 0.0, 1.0)

        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = (img * 255).astype(np.uint8)
        return img


class BramboxToTensor(BaseTransform):
    """ Converts a list of brambox annotation objects to a tensor.

    Args:
        dimension (tuple, optional): Default size of the transformed images, expressed as a (width, height) tuple; Default **None**
        dataset (lightnet.data.Dataset, optional): Dataset that uses this transform; Default **None**
        max_anno (Number, optional): Maximum number of annotations in the list; Default **50**
        class_label_map (list, optional): class label map to convert class names to an index; Default **None**

    Return:
        torch.Tensor: tensor of dimension [max_anno, 5] containing [class_idx,center_x,center_y,width,height] for every detection

    Warning:
        If no class_label_map is given, this function will first try to convert the class_label to an integer. If that fails, it is simply given the number 0.
    """
    def __init__(self, dimension=None, dataset=None, max_anno=50, class_label_map=None):
        super().__init__(dimension=dimension, dataset=dataset, max_anno=max_anno, class_label_map=class_label_map)
        if self.dimension is None and self.dataset is None:
            raise ValueError('This transform either requires a dimension or a dataset to infer the dimension')
        if self.class_label_map is None:
            log.warn('No class_label_map given. If the class_labels are not integers, they will be set to zero.')

    def __call__(self, data):
        if self.dataset is not None:
            dim = self.dataset.input_dim
        else:
            dim = self.dimension
        return self.apply(data, dim, self.max_anno, self.class_label_map)

    @classmethod
    def apply(cls, data, dimension, max_anno=None, class_label_map=None):
        if not isinstance(data, collections.Sequence):
            raise TypeError(f'BramboxToTensor only works with <brambox annotation list> [{type(data)}]')

        anno_np = np.array([cls._tf_anno(anno, dimension, class_label_map) for anno in data], dtype=np.float32)

        if max_anno is not None:
            anno_len = len(data)
            if anno_len > max_anno:
                raise ValueError(f'More annotations than maximum allowed [{anno_len}/{max_anno}]')

            z_np = np.zeros((max_anno-anno_len, 5), dtype=np.float32)
            z_np[:, 0] = -1

            if anno_len > 0:
                return torch.from_numpy(np.concatenate((anno_np, z_np)))
            else:
                return torch.from_numpy(z_np)
        else:
            return torch.from_numpy(anno_np)

    @staticmethod
    def _tf_anno(anno, dimension, class_label_map):
        """ Transforms brambox annotation to list """
        net_w, net_h = dimension

        if class_label_map is not None:
            cls = class_label_map.index(anno.class_label)
        else:
            try:
                cls = int(anno.class_label)
            except ValueError:
                cls = 0

        cx = (anno.x_top_left + (anno.width / 2)) / net_w
        cy = (anno.y_top_left + (anno.height / 2)) / net_h
        w = anno.width / net_w
        h = anno.height / net_h
        return [cls, cx, cy, w, h]
