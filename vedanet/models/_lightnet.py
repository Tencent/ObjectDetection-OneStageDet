#
#   Base lightnet network module structure
#   Copyright EAVISE
#

import logging as log
import torch
import torch.nn as nn
import time

__all__ = ['Lightnet']


class Lightnet(nn.Module):
    """ This class provides an abstraction layer on top of :class:`pytorch:torch.nn.Module` and is used as a base for every network implemented in this framework.
    There are 2 basic ways of using this class:

    - Override the ``forward()`` function.
      This makes :class:`lightnet.network.Lightnet` networks behave just like PyTorch modules.
    - Define ``self.loss`` and ``self.postprocess`` as functions and override the ``_forward()`` function.
      This class will then automatically call the loss and postprocess functions on the output of ``_forward()``,
      depending whether the network is training or evaluating.

    Attributes:
        self.seen (int): The number of images the network has processed to train *(used by engine)*

    Note:
        If you define **self.layers** as a :class:`pytorch:torch.nn.Sequential` or :class:`pytorch:torch.nn.ModuleList`,
        the default ``_forward()`` function can use these layers automatically to run the network.

    Warning:
        If you use your own ``forward()`` function, you need to update the **self.seen** parameter
        whenever the network is training.
    """
    def __init__(self):
        super().__init__()

        # Parameters
        self.layers = None
        self.loss = None
        self.postprocess = None
        self.seen = 0

    def _forward(self, x):
        log.debug('Running default forward functions')
        if isinstance(self.layers, nn.Sequential):
            return self.layers(x)
        elif isinstance(self.layers, nn.ModuleList):
            log.warn('No _forward function defined, looping sequentially over modulelist')
            for _, module in enumerate(self.layers):
                x = module(x)
            return x
        else:
            raise NotImplementedError(f'No _forward function defined and no default behaviour for this type of layers [{type(self.layers)}]')

    def forward(self, x, target=None):
        """ This default forward function will compute the output of the network as ``self._forward(x)``.
        Then, depending on whether you are training or evaluating, it will pass that output to ``self.loss()`` or ``self.posprocess()``. |br|
        This function also increments the **self.seen** variable.

        Args:
            x (torch.autograd.Variable): Input variable
            target (torch.autograd.Variable, optional): Target for the loss function; Required if training and optional otherwise (see note)

        Note:
            If you are evaluating your network and you pass a target variable, the network will return a (output, loss) tuple.
            This is usefull for testing your network, as you usually want to know the validation loss.
        """
        if self.training:
            self.seen += x.size(0)
            t1 = time.time()
            outputs = self._forward(x)
            t2 = time.time()
            
            assert len(outputs) == len(self.loss)

            loss = 0
            for idx in range(len(outputs)):
                assert callable(self.loss[idx])
                t1 = time.time()
                loss += self.loss[idx](outputs[idx], target)
                t2 = time.time()
            return loss
        else:
            outputs = self._forward(x)
            '''
            if target is not None and callable(self.loss):
                loss = self.loss(x.clone(), target)
            else:
                loss = None
            '''
            if self.postprocess is None:
                return # speed
            loss = None
            dets = []

            tdets = []
            for idx in range(len(outputs)):
                assert callable(self.postprocess[idx])
                tdets.append(self.postprocess[idx](outputs[idx]))

            batch = len(tdets[0])
            for b in range(batch):
                single_dets = []
                for op in range(len(outputs)):
                    single_dets.extend(tdets[op][b])
                dets.append(single_dets)

            if loss is not None:
                return dets, loss
            else:
                return dets, 0.0

    def modules_recurse(self, mod=None):
        """ This function will recursively loop over all module children.

        Args:
            mod (torch.nn.Module, optional): Module to loop over; Default **self**
        """
        if mod is None:
            mod = self

        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                yield from self.modules_recurse(module)
            else:
                yield module

    def init_weights(self, mode='fan_in', slope=0):
        info_list = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                info_list.append(str(m))
                nn.init.kaiming_normal_(m.weight, a=slope, mode=mode)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                info_list.append(str(m))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                info_list.append(str(m))
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        log.info('Init weights\n\n%s\n' % '\n'.join(info_list))

    def load_weights(self, weights_file, clear=False):
        """ This function will load the weights from a file.
        It also allows to load in weights file with only a part of the weights in.

        Args:
            weights_file (str): path to file
        """
        old_state = self.state_dict()
        state = torch.load(weights_file, lambda storage, loc: storage)
        self.seen = 0 if clear else state['seen']

        '''
        for key in list(state['weights'].keys()):
            if '.layer.' in key:
                log.info('Deprecated weights file found. Consider resaving your weights file before this manual intervention gets removed')
                new_key = key.replace('.layer.', '.layers.')
                state['weights'][new_key] = state['weights'].pop(key)

        new_state = state['weights']
        if new_state.keys() != old_state.keys():
            log.warn('Modules not matching, performing partial update')
            new_state = {k: v for k, v in new_state.items() if k in old_state}
            old_state.update(new_state)
            new_state = old_state
        self.load_state_dict(new_state)
        '''
        self.load_state_dict(state['weights'])

        if hasattr(self.loss, 'seen'):
            self.loss.seen = self.seen

        log.info(f'Loaded weights from {weights_file}')

    def save_weights(self, weights_file, seen=None):
        """ This function will save the weights to a file.

        Args:
            weights_file (str): path to file
            seen (int, optional): Number of images trained on; Default **self.seen**
        """
        if seen is None:
            seen = self.seen

        state = {
            'seen': seen,
            'weights': self.state_dict()
        }
        torch.save(state, weights_file)

        log.info(f'Saved weights as {weights_file}')
