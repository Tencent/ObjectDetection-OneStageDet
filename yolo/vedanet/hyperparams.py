import logging as log
import torch

__all__ = ['HyperParams']

class HyperParams(object):
    def __init__(self, config, train_flag=1):
        
        self.cuda = True
        self.labels = config['labels']
        self.classes = len(self.labels)
        self.data_root = config['data_root_dir'] 
        self.model_name = config['model_name']

        # cuda check
        if self.cuda:
            if not torch.cuda.is_available():
                log.debug('CUDA not available')
                self.cuda = False
            else:
                log.debug('CUDA enabled')

        if train_flag == 1:
            cur_cfg = config

            self.nworkers = cur_cfg['nworkers'] 
            self.pin_mem = cur_cfg['pin_mem'] 
            dataset = cur_cfg['dataset']
            self.trainfile = f'{self.data_root}/{dataset}.pkl'

            self.network_size = cur_cfg['input_shape']

            self.batch = cur_cfg['batch_size']
            self.mini_batch = cur_cfg['mini_batch_size']
            self.max_batches = cur_cfg['max_batches']

            self.jitter = 0.3
            self.flip = 0.5
            self.hue = 0.1
            self.sat = 1.5
            self.val = 1.5

            self.learning_rate = cur_cfg['warmup_lr'] 
            self.momentum = cur_cfg['momentum']
            self.decay = cur_cfg['decay'] 
            self.lr_steps = cur_cfg['lr_steps']
            self.lr_rates = cur_cfg['lr_rates'] 

            self.backup = cur_cfg['backup_interval']
            self.bp_steps = cur_cfg['backup_steps']
            self.bp_rates = cur_cfg['backup_rates']
            self.backup_dir = cur_cfg['backup_dir']

            self.resize = cur_cfg['resize_interval'] 
            self.rs_steps = []
            self.rs_rates = []

            self.weights = cur_cfg['weights']
            self.clear = cur_cfg['clear']
        elif train_flag == 2:
            cur_cfg = config

            dataset = cur_cfg['dataset']
            self.testfile = f'{self.data_root}/{dataset}.pkl'
            self.nworkers = cur_cfg['nworkers'] 
            self.pin_mem = cur_cfg['pin_mem'] 
            self.network_size = cur_cfg['input_shape']
            self.batch = cur_cfg['batch_size']
            self.weights = cur_cfg['weights']
            self.conf_thresh = cur_cfg['conf_thresh']
            self.nms_thresh = cur_cfg['nms_thresh']
            self.results = cur_cfg['results']

        else:
            cur_cfg = config

            self.network_size = cur_cfg['input_shape']
            self.batch = cur_cfg['batch_size']
            self.max_iters = cur_cfg['max_iters']

