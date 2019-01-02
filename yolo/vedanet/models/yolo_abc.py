#
#   Darknet YOLOv2 model
#   Copyright EAVISE
#

import os
from collections import OrderedDict, Iterable
import torch
import torch.nn as nn
from .. import data as vnd
from ._darknet import Darknet

__all__ = ['YoloABC']


class YoloABC(Darknet):
    def __init__(self):
        """ Network initialisation """
        super().__init__()

        # Parameters
        self.num_classes = None
        self.anchors = None
        self.anchors_mask = None
        self.nloss = None
        self.loss = None
        self.postprocess = None
        self.train_flag = None # 1 for train, 2 for test, 0 for speed
        self.test_args = None

        # Network
        # backbone
        #self.backbone = backbone() 
        # head
        #self.head = head()

        #if weights_file is not None:
        #    self.load_weights(weights_file, clear)

    def _forward(self, x):
        pass

    def compose(self, x, features, loss_fn):
        """
        generate loss and postprocess
        """
        if self.train_flag == 1: # train
            if self.loss is None:
                self.loss = [] # for training

                for idx in range(self.nloss):
                    reduction = float(x.shape[2] / features[idx].shape[2]) # n, c, h, w
                    self.loss.append(loss_fn(self.num_classes, self.anchors, self.anchors_mask[idx],
                        reduction, self.seen, head_idx=idx))
        elif self.train_flag == 2: # test
            if self.postprocess is None:
                self.postprocess = [] # for testing

                conf_thresh = self.test_args['conf_thresh']
                network_size = self.test_args['network_size']
                labels = self.test_args['labels']
                for idx in range(self.nloss):
                    reduction = float(x.shape[2] / features[idx].shape[2]) # n, c, h, w
                    cur_anchors = [self.anchors[ii] for ii in self.anchors_mask[idx]]
                    cur_anchors = [(ii[0] / reduction, ii[1] / reduction) for ii in cur_anchors] # abs to relative
                    self.postprocess.append(vnd.transform.Compose([
                        vnd.transform.GetBoundingBoxes(self.num_classes, cur_anchors, conf_thresh),
                        vnd.transform.TensorToBrambox(network_size, labels)
                        ]))
        # else, speed
