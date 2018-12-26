#
#   Base engine class
#   Copyright EAVISE
#

import sys
import logging as log
import signal
from statistics import mean
from abc import ABC, abstractmethod
import torch

import vedanet as vn

__all__ = ['Engine']


class Engine(ABC):
    """ This class removes the boilerplate code needed for writing your training cycle. |br|
    Here is the code that runs when the engine is called:

    .. literalinclude:: /../lightnet/engine/engine.py
       :language: python
       :pyobject: Engine.__call__
       :dedent: 4

    Args:
        network (lightnet.network.Darknet, optional): Lightnet network to train
        optimizer (torch.optim, optional): Optimizer for the network
        dataloader (lightnet.data.DataLoader or torch.utils.data.DataLoader, optional): Dataloader for the training data
        **kwargs (dict, optional): Keywords arguments that will be set as attributes of the engine

    Attributes:
        self.network: Lightnet network
        self.optimizer: Torch optimizer
        self.batch_size: Number indicating batch_size; Default **1**
        self.mini_batch_size: Size of a mini_batch; Default **1**
        self.max_batches: Maximum number of batches to process; Default **None**
        self.test_rate: How often to run test; Default **None**
        self.sigint: Boolean value indicating whether a SIGINT (CTRL+C) was send; Default **False**
    """
    __allowed_overwrite = ['batch_size', 'mini_batch_size', 'max_batches', 'test_rate']
    batch_size = 1
    mini_batch_size = 1
    max_batches = None
    test_rate = None

    #def __init__(self, network, optimizer, dataloader, **kwargs):
    def __init__(self, network, optimizer, dataloader):
        if network is not None:
            self.network = network
        else:
            log.warn('No network given, make sure to have a self.network property for this engine to work with.')

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            log.warn('No optimizer given, make sure to have a self.optimizer property for this engine to work with.')

        if dataloader is not None:
            self.dataloader = dataloader
        else:
            log.warn('No dataloader given, make sure to have a self.dataloader property for this engine to work with.')

        # Rates
        self.__lr = self.optimizer.param_groups[0]['lr']
        self.__rates = {}

        # Sigint handling
        self.sigint = False
        signal.signal(signal.SIGINT, self.__sigint_handler)

        # Set attributes
        '''
        for key in kwargs:
            if not hasattr(self, key) or key in self.__allowed_overwrite:
                setattr(self, key, kwargs[key])
            else:
                log.warn(f'{key} attribute already exists on engine. Keeping original value [{getattr(self, key)}]')
        '''

    def __call__(self):
        """ Start the training cycle. """
        self.start()
        self._update_rates()
        if self.test_rate is not None:
            last_test = self.batch - (self.batch % self.test_rate)

        log.info('Start training')
        self.network.train()
        while True:
            loader = self.dataloader
            for idx, data in enumerate(loader):
                # Forward and backward on (mini-)batches
                self.process_batch(data)
                if (idx + 1) % self.batch_subdivisions != 0:
                    continue

                # Optimizer step
                self.train_batch()

                # Check if we need to stop training
                if self.quit() or self.sigint:
                    log.info('Reached quitting criteria')
                    return

                # Check if we need to perform testing
                if self.test_rate is not None and self.batch - last_test >= self.test_rate:
                    log.info('Start testing')
                    last_test += self.test_rate
                    self.network.eval()
                    self.test()
                    log.debug('Done testing')
                    self.network.train()

                # Check if we need to stop training
                if self.quit() or self.sigint:
                    log.info('Reached quitting criteria')
                    return

                # Automatically update registered rates
                self._update_rates()

                # Not enough mini-batches left to have an entire batch
                if (len(loader) - idx) <= self.batch_subdivisions:
                    break

    @property
    def batch(self):
        """ Get current batch number.

        Return:
            int: Computed as self.network.seen // self.batch_size
        """
        return self.network.seen // self.batch_size

    @property
    def batch_subdivisions(self):
        """ Get number of mini-batches per batch.

        Return:
            int: Computed as self.batch_size // self.mini_batch_size
        """
        return self.batch_size // self.mini_batch_size

    @property
    def learning_rate(self):
        """ Get and set the learning rate

        Args:
            lr (Number): Set the learning rate for all values of optimizer.param_groups[i]['lr']

        Return:
            Number: The current learning rate
        """
        return self.__lr

    @learning_rate.setter
    def learning_rate(self, lr):
        log.info(f'Adjusting learning rate to [{lr*self.batch_size}]')
        self.__lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def add_rate(self, name, steps, values, default=None):
        """ Add a rate to the engine.
        Rates are object attributes that automatically change according to the current batch number.

        Args:
            name (str): Name that will be used for the attribute. You can access the value with self.name
            steps (list): Batches at which the rate should change
            values (list): New values that will be used for the attribute
            default (optional): Default value to use for the rate; Default **None**

        Note:
            You can also set the ``learning_rate`` with this method.
            This will actually use the ``learning_rate`` computed property of this class and set the learning rate of the optimizer. |br|
            This is great for automating adaptive learning rates, and can work in conjunction with pytorch schedulers.

        Example:
            >>> class MyEngine(ln.engine.Engine):
            ...     batch_size = 2
            ...     def process_batch(self, data):
            ...         raise NotImplementedError()
            ...     def train_batch(self):
            ...         raise NotImplementedError()
            >>> net = ln.models.Yolo()
            >>> eng = MyEngine(
            ...     net,
            ...     torch.optim.SGD(net.parameters(), lr=.1),
            ...     None    # Should be dataloader
            ... )
            >>> eng.add_rate('test_rate', [1000, 10000], [100, 500], 50)
            >>> eng.add_rate('learning_rate', [1000, 10000], [.01, .001])
            >>> eng.test_rate
            50
            >>> eng.learning_rate
            0.1
            >>> net.seen = 2000     # batch_size = 2
            >>> eng._update_rates() # Happens automatically during training loop
            >>> eng.test_rate
            100
            >>> eng.learning_rate
            0.01
        """
        if default is not None or not hasattr(self, name):
            setattr(self, name, default)
        if name in self.__rates:
            log.warn(f'{name} rate was already used, overwriting...')

        if len(steps) > len(values):
            diff = len(steps) - len(values)
            values = values + diff * [values[-1]]
            log.warn(f'{name} has more steps than values, extending values to {values}')
        elif len(steps) < len(values):
            values = values[:len(steps)]
            log.warn(f'{name} has more values than steps, shortening values to {values}')

        self.__rates[name] = (steps, values)

    def _update_rates(self):
        """ Update rates according to batch size. |br|
        This function gets automatically called every batch, and should generally not be called by the user.
        """
        for key, (steps, values) in self.__rates.items():
            new_rate = None
            for i in range(len(steps)):
                if self.batch >= steps[i]:
                    new_rate = values[i]
                else:
                    break

            if new_rate is not None and new_rate != getattr(self, key):
                #log.info(f'Adjusting {key} [{new_rate*self.batch_size}]')
                setattr(self, key, new_rate)

    def start(self):
        """ First function that gets called when starting the engine. |br|
            Use it to create your dataloader, set the correct starting values for your rates, etc.
        """
        pass

    @abstractmethod
    def process_batch(self, data):
        """ This function should contain the code to process the forward and backward pass of one (mini-)batch. """
        pass

    @abstractmethod
    def train_batch(self):
        """ This function should contain the code to update the weights of the network. |br|
        Statistical computations, performing backups at regular intervals, etc. also happen here.
        """
        pass

    def test(self):
        """ This function should contain the code to perform an evaluation on your test-set. """
        log.error('test() function is not implemented')

    def quit(self):
        """ This function gets called after every training epoch and decides if the training cycle continues.

        Return:
            Boolean: Whether are not to stop the training cycle

        Note:
            This function gets called before checking the ``self.sigint`` attribute.
            This means you can also check this attribute in this function. |br|
            If it evaluates to **True**, you know the program will exit after this function and you can thus
            perform the necessary actions (eg. save final weights).
        """
        if self.max_batches is not None:
            return self.batch >= self.max_batches
        else:
            return False

    def __sigint_handler(self, signal, frame):
        if not self.sigint:
            log.debug('SIGINT caught. Waiting for gracefull exit')
            self.sigint = True
