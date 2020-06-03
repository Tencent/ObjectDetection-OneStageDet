#
#   Lightnet related postprocessing
#   Thers are functions to transform the output of the network to brambox detection objects
#   Copyright EAVISE
#


# modified by mileistone

import logging as log
import torch
from torch.autograd import Variable
from brambox.boxes.detections.detection import *
from .util import BaseTransform

__all__ = ['GetBoundingBoxes', 'NonMaxSupression', 'TensorToBrambox', 'ReverseLetterbox']


class GetBoundingBoxes(BaseTransform):
    """ Convert output from darknet networks to bounding box tensor.

    Args:
        num_classes (int): number of categories
        anchors (list): 2D list representing anchor boxes (see :class:`lightnet.network.Darknet`)
        conf_thresh (Number [0-1]): Confidence threshold to filter detections

    Returns:
        (list [Batch x Tensor [Boxes x 6]]): **[x_center, y_center, width, height, confidence, class_id]** for every bounding box

    Note:
        The output tensor uses relative values for its coordinates.
    """
    def __init__(self, num_classes, anchors, conf_thresh):
        super().__init__(num_classes=num_classes, anchors=anchors, conf_thresh=conf_thresh)

    @classmethod
    def apply(cls, network_output, num_classes, anchors, conf_thresh):
        num_anchors = len(anchors)
        anchor_step = len(anchors[0])
        anchors = torch.Tensor(anchors)
        if isinstance(network_output, Variable):
            network_output = network_output.data

        # Check dimensions
        if network_output.dim() == 3:
            network_output.unsqueeze_(0)

        # Variables
        cuda = network_output.is_cuda
        batch = network_output.size(0)
        h = network_output.size(2)
        w = network_output.size(3)

        # Compute xc,yc, w,h, box_score on Tensor
        lin_x = torch.linspace(0, w-1, w).repeat(h, 1).view(h*w)
        lin_y = torch.linspace(0, h-1, h).repeat(w, 1).t().contiguous().view(h*w)
        anchor_w = anchors[:, 0].contiguous().view(1, num_anchors, 1)
        anchor_h = anchors[:, 1].contiguous().view(1, num_anchors, 1)
        if cuda:
            lin_x = lin_x.cuda()
            lin_y = lin_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        network_output = network_output.view(batch, num_anchors, -1, h*w)   # -1 == 5+num_classes (we can drop feature maps if 1 class)
        network_output[:, :, 0, :].sigmoid_().add_(lin_x).div_(w)           # X center
        network_output[:, :, 1, :].sigmoid_().add_(lin_y).div_(h)           # Y center
        network_output[:, :, 2, :].exp_().mul_(anchor_w).div_(w)            # Width
        network_output[:, :, 3, :].exp_().mul_(anchor_h).div_(h)            # Height
        network_output[:, :, 4, :].sigmoid_()                               # Box score

        conf_scores = network_output[:, :, 4, :] ## mileistone

        # Compute class_score
        if num_classes > 1:
            if torch.__version__.startswith('0.3'):
                cls_scores = torch.nn.functional.softmax(Variable(network_output[:, :, 5:, :], volatile=True), 2).data
            else:
                with torch.no_grad():
                    cls_scores = torch.nn.functional.softmax(network_output[:, :, 5:, :], 2) 
                    cls_scores = (cls_scores * conf_scores.unsqueeze(2).expand_as(cls_scores)).transpose(2,3)
                    cls_scores = cls_scores.contiguous().view(cls_scores.size(0), cls_scores.size(1), -1)
        else:
            cls_scores = network_output[:, :, 4, :]
            #cls_max = network_output[:, :, 4, :]
            #cls_max_idx = torch.zeros_like(cls_max)

        score_thresh = cls_scores > conf_thresh
        score_thresh_flat = score_thresh.view(-1)

        if score_thresh.sum() == 0:
            boxes = []
            for i in range(batch):
                boxes.append(torch.Tensor([]))
            return boxes

        # Mask select boxes > conf_thresh
        coords = network_output.transpose(2, 3)[..., 0:4]
        coords = coords.unsqueeze(3).expand(coords.size(0),coords.size(1),coords.size(2), 
                num_classes,coords.size(3)).contiguous().view(coords.size(0),coords.size(1),-1,coords.size(3))
        coords = coords[score_thresh[..., None].expand_as(coords)].view(-1, 4)
        scores = cls_scores[score_thresh].view(-1, 1)
        idx = (torch.arange(num_classes)).repeat(batch, num_anchors, w*h).cuda()
        idx = idx[score_thresh].view(-1, 1).float()
        detections = torch.cat([coords, scores, idx], dim=1)

        # Get indexes of splits between images of batch
        max_det_per_batch = num_anchors * h * w * num_classes
        slices = [slice(max_det_per_batch * i, max_det_per_batch * (i+1)) for i in range(batch)]
        det_per_batch = torch.IntTensor([score_thresh_flat[s].int().sum() for s in slices])
        split_idx = torch.cumsum(det_per_batch, dim=0)

        # Group detections per image of batch
        boxes = []
        start = 0
        for end in split_idx:
            boxes.append(detections[start: end])
            start = end

        return boxes


class NonMaxSupression(BaseTransform):
    """ Performs nms on the bounding boxes, filtering boxes with a high overlap.

    Args:
        nms_thresh (Number [0-1]): Overlapping threshold to filter detections with non-maxima suppresion
        class_nms (Boolean, optional): Whether to perform nms per class; Default **True**
        fast (Boolean, optional): This flag can be used to select a much faster variant on the algorithm, that suppresses slightly more boxes; Default **False**

    Returns:
        (list [Batch x Tensor [Boxes x 6]]): **[x_center, y_center, width, height, confidence, class_id]** for every bounding box

    Note:
        This post-processing function expects the input to be bounding boxes,
        like the ones created by :class:`lightnet.data.GetBoundingBoxes` and outputs exactly the same format.
    """
    def __init__(self, nms_thresh, class_nms=True, fast=False):
        super().__init__(nms_thresh=nms_thresh, class_nms=class_nms, fast=fast)

    @classmethod
    def apply(cls, boxes, nms_thresh, class_nms=True, fast=False):
        return [cls._nms(box, nms_thresh, class_nms, fast) for box in boxes]

    @staticmethod
    def _nms(boxes, nms_thresh, class_nms, fast):
        """ Non maximum suppression.

        Args:
          boxes (tensor): Bounding boxes of one image

        Return:
          (tensor): Pruned boxes
        """
        if boxes.numel() == 0:
            return boxes
        cuda = boxes.is_cuda

        a = boxes[:, :2]
        b = boxes[:, 2:4]
        bboxes = torch.cat([a-b/2, a+b/2], 1)
        scores = boxes[:, 4]
        classes = boxes[:, 5]

        # Sort coordinates by descending score
        scores, order = scores.sort(0, descending=True)
        x1, y1, x2, y2 = bboxes[order].split(1, 1)

        # Compute dx and dy between each pair of boxes (these mat contain every pair twice...)
        dx = (x2.min(x2.t()) - x1.max(x1.t())).clamp(min=0)
        dy = (y2.min(y2.t()) - y1.max(y1.t())).clamp(min=0)

        # Compute iou
        intersections = dx * dy
        areas = (x2 - x1) * (y2 - y1)
        unions = (areas + areas.t()) - intersections
        ious = intersections / unions

        # Filter based on iou (and class)
        conflicting = (ious > nms_thresh).triu(1)

        if class_nms:
            same_class = (classes.unsqueeze(0) == classes.unsqueeze(1))
            conflicting = (conflicting & same_class)

        keep = conflicting.sum(0).byte()
        if not fast:        # How can we optimize this part!?
            keep = keep.cpu()
            conflicting = conflicting.cpu()

            keep_len = len(keep) - 1
            for i in range(1, keep_len):
                if keep[i] > 0:
                    keep -= conflicting[i]

            if cuda:
                keep = keep.cuda()

        keep = (keep == 0)
        return boxes[order][keep[:, None].expand_as(boxes)].view(-1, 6).contiguous()


class TensorToBrambox(BaseTransform):
    """ Converts a tensor to a list of brambox objects.

    Args:
        network_size (tuple): Tuple containing the width and height of the images going in the network
        class_label_map (list, optional): List of class labels to transform the class id's in actual names; Default **None**

    Returns:
        (list [list [brambox.boxes.Detection]]): list of brambox detections per image

    Note:
        If no `class_label_map` is given, this transform will simply convert the class id's in a string.

    Note:
        Just like everything in PyTorch, this transform only works on batches of images.
        This means you need to wrap your tensor of detections in a list if you want to run this transform on a single image.
    """
    def __init__(self, network_size, class_label_map=None):
        super().__init__(network_size=network_size, class_label_map=class_label_map)
        if self.class_label_map is None:
            log.warn('No class_label_map given. The indexes will be used as class_labels.')

    @classmethod
    def apply(cls, boxes, network_size, class_label_map=None):
        converted_boxes = []
        for box in boxes:
            if box.nelement() == 0:
                converted_boxes.append([])
            else:
                converted_boxes.append(cls._convert(box, network_size[0], network_size[1], class_label_map))
        return converted_boxes

    @staticmethod
    def _convert(boxes, width, height, class_label_map):
        boxes[:, 0:3:2].mul_(width)
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1:4:2].mul_(height)
        boxes[:, 1] -= boxes[:, 3] / 2

        brambox = []
        for box in boxes:
            det = Detection()
            if torch.__version__.startswith('0.3'):
                det.x_top_left = box[0]
                det.y_top_left = box[1]
                det.width = box[2]
                det.height = box[3]
                det.confidence = box[4]
                if class_label_map is not None:
                    det.class_label = class_label_map[int(box[5])]
                else:
                    det.class_label = str(int(box[5]))
            else:
                det.x_top_left = box[0].item()
                det.y_top_left = box[1].item()
                det.width = box[2].item()
                det.height = box[3].item()
                det.confidence = box[4].item()
                if class_label_map is not None:
                    det.class_label = class_label_map[int(box[5].item())]
                else:
                    det.class_label = str(int(box[5].item()))

            brambox.append(det)

        return brambox


class ReverseLetterbox(BaseTransform):
    """ Performs a reverse letterbox operation on the bounding boxes, so they can be visualised on the original image.

    Args:
        network_size (tuple): Tuple containing the width and height of the images going in the network
        image_size (tuple): Tuple containing the width and height of the original images

    Returns:
        (list [list [brambox.boxes.Detection]]): list of brambox detections per image

    Note:
        This transform works on :class:`brambox.boxes.Detection` objects,
        so you need to apply the :class:`~lightnet.data.TensorToBrambox` transform first.

    Note:
        Just like everything in PyTorch, this transform only works on batches of images.
        This means you need to wrap your tensor of detections in a list if you want to run this transform on a single image.
    """
    def __init__(self, network_size, image_size):
        super().__init__(network_size=network_size, image_size=image_size)

    @classmethod
    def apply(cls, boxes, network_size, image_size):
        im_w, im_h = image_size[:2]
        net_w, net_h = network_size[:2]

        if im_w == net_w and im_h == net_h:
            scale = 1
        elif im_w / net_w >= im_h / net_h:
            scale = im_w/net_w
        else:
            scale = im_h/net_h
        pad = int((net_w - im_w/scale) / 2), int((net_h - im_h/scale) / 2)

        converted_boxes = []
        for b in boxes:
            converted_boxes.append(cls._transform(b, scale, pad))
        return converted_boxes

    @staticmethod
    def _transform(boxes, scale, pad):
        for box in boxes:
            box.x_top_left -= pad[0]
            box.y_top_left -= pad[1]

            box.x_top_left *= scale
            box.y_top_left *= scale
            box.width *= scale
            box.height *= scale
        return boxes
