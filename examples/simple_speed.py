import time
import torch
import os

import sys
sys.path.insert(0, '.')

import vedanet as vn

__all__ = ['speed']


def speed():
    print('Creating network')

    batch = 1
    gpus = '7'
    network_size = (544, 544)
    max_iters = 200

    use_cuda = True if gpus is not None else False
    if gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    net =  vn.network.backbone.Mobilenet()
    net.eval()
    print('Net structure\n%s' % net)

    if use_cuda:
        net.cuda()

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

    print('Average %.3fms per forward in %d iteration (batch size %d, shape %dx%d)' % 
            (1000 * elapse / max_iters, max_iters, batch, network_size[1], network_size[0]))


if __name__ == '__main__':
    speed()
