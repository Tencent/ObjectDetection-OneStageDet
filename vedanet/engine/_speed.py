import logging as log
import time
import torch
from torchvision import transforms as tf
from statistics import mean
import os

from .. import data as vn_data
from .. import models
from . import engine
from utils.test import voc_wrapper

__all__ = ['speed']


def speed(hyper_params):
    log.debug('Creating network')

    model_name = hyper_params.model_name
    batch = hyper_params.batch
    use_cuda = hyper_params.cuda
    network_size = hyper_params.network_size
    max_iters = hyper_params.max_iters

    net = models.__dict__[model_name](hyper_params.classes, train_flag=0)
    net.eval()
    print('Net structure\n%s' % net)

    if use_cuda:
        net.cuda()

    log.debug('Running network')

    data = torch.randn(batch, 3, network_size[1], network_size[0], dtype=torch.float)
    if use_cuda:
        data = data.cuda()

    torch.cuda.synchronize()
    start_time = time.time()
    for idx in range(max_iters):
        with torch.no_grad():
            net(data)
    torch.cuda.synchronize()
    end_time = time.time()
    elapse = (end_time - start_time)

    print('%s: Average %.3fms per forward in %d iteration (batch size %d, shape %dx%d)' % 
            (model_name, 1000 * elapse / max_iters, max_iters, batch, network_size[0], network_size[1]))


