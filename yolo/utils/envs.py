import sys
import os
import copy
from datetime import datetime
import logging
import torch
import random
import numpy as np


# individual packages
from .fileproc import safeMakeDirs
from .cfg_parser import getConfig


def setLogging(log_dir, stdout_flag):
    dt = datetime.now()
    log_name = dt.strftime('%Y-%m-%d_time_%H_%M_%S') + '.log'

    log_fp = os.path.join(log_dir, log_name)
    #print os.path.abspath(log_fp)

    if stdout_flag:
        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(filename=log_fp, format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)


def combineConfig(cur_cfg, train):
    ret_cfg = {}
    for k, v in cur_cfg.items():
        if k == 'train' or k == 'test':
            continue
        ret_cfg[k] = v
    key = 'train' if train else 'test'
    for k, v in cur_cfg[key].items():
        ret_cfg[k] = v
    return ret_cfg


def initEnv(train, model_name):
    cfgs_root = 'cfgs'
    cur_cfg = getConfig(cfgs_root, model_name)

    root_dir = cur_cfg['output_root']
    cur_cfg['model_name'] = model_name
    version = cur_cfg['output_version']
    work_dir = os.path.join(root_dir, model_name, version)

    backup_name = cur_cfg['backup_name']
    log_name = cur_cfg['log_name']

    backup_dir = os.path.join(work_dir, backup_name)
    log_dir = os.path.join(work_dir, log_name)

    safeMakeDirs(backup_dir)
    safeMakeDirs(log_dir)

    if train:
        stdout_flag = cur_cfg['train']['stdout']
        setLogging(log_dir, stdout_flag)

        gpus = cur_cfg['train']['gpus']
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus

        cur_cfg['train']['backup_dir'] = backup_dir
    else:
        stdout_flag = cur_cfg['test']['stdout']
        setLogging(log_dir, stdout_flag)

        gpus = cur_cfg['test']['gpus']
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    ret_cfg = combineConfig(cur_cfg, train)

    return ret_cfg


def randomSeeding(seed):
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)


if __name__ == '__main__':
    pass
