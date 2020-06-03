import os
import argparse
import logging as log
import time
from statistics import mean
import numpy as np
import torch
from torchvision import transforms as tf
from pprint import pformat

import sys
sys.path.insert(0, '.')

import brambox.boxes as bbb
import vedanet as vn
from utils.envs import initEnv, randomSeeding


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='OneDet: an one stage framework based on PyTorch')
    parser.add_argument('model_name', help='model name', default=None)
    args = parser.parse_args()

    train_flag = 1
    config = initEnv(train_flag=train_flag, model_name=args.model_name)
    #randomSeeding(0)

    log.info('Config\n\n%s\n' % pformat(config))

    # init env
    hyper_params = vn.hyperparams.HyperParams(config, train_flag=train_flag)

    # int eng
    eng = vn.engine.VOCTrainingEngine(hyper_params)

    # run eng
    b1 = eng.batch
    t1 = time.time()
    eng()
    t2 = time.time()
    b2 = eng.batch

    log.info(f'\nDuration of {b2-b1} batches: {t2-t1} seconds [{round((t2-t1)/(b2-b1), 3)} sec/batch]')
