import logging as log
import torch
from torchvision import transforms as tf
from statistics import mean
import os

from .. import data
from .. import models
from . import engine

__all__ = ['VOCTrainingEngine']

class VOCDataset(data.BramboxDataset):
    def __init__(self, hyper_params):
        anno = hyper_params.trainfile
        root = hyper_params.data_root
        flip = hyper_params.flip
        jitter = hyper_params.jitter
        hue, sat, val = hyper_params.hue, hyper_params.sat ,hyper_params.val
        network_size = hyper_params.network_size
        labels = hyper_params.labels

        rf  = data.transform.RandomFlip(flip)
        rc  = data.transform.RandomCropLetterbox(self, jitter)
        hsv = data.transform.HSVShift(hue, sat, val)
        it  = tf.ToTensor()

        img_tf = data.transform.Compose([rc, rf, hsv, it])
        anno_tf = data.transform.Compose([rc, rf])

        def identify(img_id):
            #return f'{root}/VOCdevkit/{img_id}.jpg'
            return f'{img_id}'

        super(VOCDataset, self).__init__('anno_pickle', anno, network_size, labels, identify, img_tf, anno_tf)


class VOCTrainingEngine(engine.Engine):
    """ This is a custom engine for this training cycle """

    def __init__(self, hyper_params):
        self.hyper_params = hyper_params
        # all in args
        self.batch_size = hyper_params.batch
        self.mini_batch_size = hyper_params.mini_batch
        self.max_batches = hyper_params.max_batches

        self.classes = hyper_params.classes

        self.cuda = hyper_params.cuda
        self.backup_dir = hyper_params.backup_dir

        log.debug('Creating network')
        model_name = hyper_params.model_name
        net = models.__dict__[model_name](hyper_params.classes, hyper_params.weights, train_flag=1, clear=hyper_params.clear)
        log.info('Net structure\n\n%s\n' % net)
        if self.cuda:
            net.cuda()

        log.debug('Creating optimizer')
        learning_rate = hyper_params.learning_rate
        momentum = hyper_params.momentum
        decay = hyper_params.decay
        batch = hyper_params.batch
        log.info(f'Adjusting learning rate to [{learning_rate}]')
        optim = torch.optim.SGD(net.parameters(), lr=learning_rate/batch, momentum=momentum, dampening=0, weight_decay=decay*batch)

        log.debug('Creating dataloader')
        dataset = VOCDataset(hyper_params)
        dataloader = data.DataLoader(
            dataset,
            batch_size = self.mini_batch_size,
            shuffle = True,
            drop_last = True,
            num_workers = hyper_params.nworkers if self.cuda else 0,
            pin_memory = hyper_params.pin_mem if self.cuda else False,
            collate_fn = data.list_collate,
        )

        super(VOCTrainingEngine, self).__init__(net, optim, dataloader)

        self.nloss = self.network.nloss

        self.train_loss = [{'tot': [], 'coord': [], 'conf': [], 'cls': []} for _ in range(self.nloss)]

    def start(self):
        log.debug('Creating additional logging objects')
        hyper_params = self.hyper_params

        lr_steps = hyper_params.lr_steps
        lr_rates = hyper_params.lr_rates

        bp_steps = hyper_params.bp_steps
        bp_rates = hyper_params.bp_rates
        backup = hyper_params.backup

        rs_steps = hyper_params.rs_steps
        rs_rates = hyper_params.rs_rates
        resize = hyper_params.resize

        self.add_rate('learning_rate', lr_steps, [lr/self.batch_size for lr in lr_rates])
        self.add_rate('backup_rate', bp_steps, bp_rates, backup)
        self.add_rate('resize_rate', rs_steps, rs_rates, resize)

        self.dataloader.change_input_dim()

    def process_batch(self, data):
        data, target = data
        # to(device)
        if self.cuda:
            data = data.cuda()
        #data = torch.autograd.Variable(data, requires_grad=True)

        loss = self.network(data, target)
        loss.backward()

        for ii in range(self.nloss):
            self.train_loss[ii]['tot'].append(self.network.loss[ii].loss_tot.item() / self.mini_batch_size)
            self.train_loss[ii]['coord'].append(self.network.loss[ii].loss_coord.item() / self.mini_batch_size)
            self.train_loss[ii]['conf'].append(self.network.loss[ii].loss_conf.item() / self.mini_batch_size)
            if self.network.loss[ii].loss_cls is not None:
                self.train_loss[ii]['cls'].append(self.network.loss[ii].loss_cls.item() / self.mini_batch_size)
    
    def train_batch(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

        all_tot = 0.0
        all_coord = 0.0
        all_conf = 0.0
        all_cls = 0.0
        for ii in range(self.nloss):
            tot = mean(self.train_loss[ii]['tot'])
            coord = mean(self.train_loss[ii]['coord'])
            conf = mean(self.train_loss[ii]['conf'])
            all_tot += tot
            all_coord += coord
            all_conf += conf
            if self.classes > 1:
                cls = mean(self.train_loss[ii]['cls'])
                all_cls += cls

            if self.classes > 1:
                log.info(f'{self.batch} # {ii}: Loss:{round(tot, 5)} (Coord:{round(coord, 2)} Conf:{round(conf, 2)} Cls:{round(cls, 2)})')
            else:
                log.info(f'{self.batch} # {ii}: Loss:{round(tot, 5)} (Coord:{round(coord, 2)} Conf:{round(conf, 2)})')

        if self.classes > 1:
            log.info(f'{self.batch} # All : Loss:{round(all_tot, 5)} (Coord:{round(all_coord, 2)} Conf:{round(all_conf, 2)} Cls:{round(all_cls, 2)})')
        else:
            log.info(f'{self.batch} # All : Loss:{round(all_tot, 5)} (Coord:{round(all_coord, 2)} Conf:{round(all_conf, 2)})')
        self.train_loss = [{'tot': [], 'coord': [], 'conf': [], 'cls': []} for _ in range(self.nloss)]
        if self.batch % self.backup_rate == 0:
            self.network.save_weights(os.path.join(self.backup_dir, f'weights_{self.batch}.pt'))

        if self.batch % 100 == 0:
            self.network.save_weights(os.path.join(self.backup_dir, f'backup.pt'))

        if self.batch % self.resize_rate == 0:
            if self.batch + 200 >= self.max_batches:
                finish_flag = True
            else:
                finish_flag = False
            self.dataloader.change_input_dim(finish=finish_flag)

    def quit(self):
        if self.sigint:
            self.network.save_weights(os.path.join(self.backup_dir, f'backup.pt'))
            return True
        elif self.batch >= self.max_batches:
            self.network.save_weights(os.path.join(self.backup_dir, f'final.dw'))
            return True
        else:
            return False
