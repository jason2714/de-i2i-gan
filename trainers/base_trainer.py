import math
from collections import defaultdict

import torch
from torch import optim

from models import create_model
import numpy as np
from torch.cuda.amp import GradScaler


class BaseTrainer:
    """
    Trainer receives the options and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt

        # initial model
        self.model = create_model(opt)
        if opt.continue_training:
            self.model.load('latest')
        elif opt.load_model_name is not None:
            self.model.load(opt.which_epoch)
        else:
            self.model.init_weights()

        # initial losses and metrics
        self.losses = defaultdict(list)
        self.dis_outputs = defaultdict(list)
        if opt.phase == 'val':
            self.metrics = dict()

        # initial epochs and iters
        self.iter_record_path = opt.ckpt_dir / opt.name / 'iter.txt'
        self.first_epoch = 1
        self.iters = 0
        assert hasattr(self.opt, 'iters_per_epoch'), 'opt must have attribute {iters_per_epoch}, ' \
                                                     'it can be calculated by length of loader'
        if opt.continue_training:
            self.first_epoch, self.iters = np.loadtxt(self.iter_record_path, delimiter=',', dtype=int)
        if self.opt.num_epochs == -1:
            self.opt.num_epochs = math.ceil(self.opt.num_iters / (self.opt.iters_per_epoch + 1e-12))
        self.opt.num_iters = self.opt.num_epochs * self.opt.iters_per_epoch
        assert self.first_epoch < self.opt.num_epochs, f'first_epoch {self.first_epoch} should not larger than ' \
                                                       f'num_epochs {self.opt.num_epochs}'
        assert self.iters < self.opt.num_iters, f'iters {self.iters} should not larger than ' \
                                                f'num_iters {self.opt.num_iters}'
        self.opt.first_epoch = self.first_epoch

        # initial optimizer and scheduler
        self._init_lr(opt)
        self._create_optimizer(opt)
        self._create_scheduler(opt)

        # initial amp
        # Creates a GradScaler once at the beginning of training.
        self.scaler = GradScaler()

    def _init_lr(self, opt):
        """transform lr from list to scalar or dict
            should be overwrite by child"""
        self.lr = opt.lr[0]

    def _create_optimizer(self, opt):
        assert isinstance(self.lr, (int, float, dict)), 'type of lr should be scalar or dict'
        optim_args = dict()
        if opt.optimizer == 'sgd':
            optim_cls = optim.SGD
        elif opt.optimizer == 'rmsprop':
            optim_cls = optim.RMSprop
        elif opt.optimizer == 'adam':
            optim_cls = optim.Adam
            optim_args['betas'] = (0.5, 0.999)
        elif opt.optimizer == 'adamw':
            optim_cls = optim.AdamW
            optim_args['betas'] = (0.9, 0.95)
        else:
            raise NameError(f'optimizer named {opt.optimizer} not defined')
        self.optimizers = {}
        for network_name, network in self.model.networks.items():
            if isinstance(self.lr, dict):
                optim_args['lr'] = self.lr[network_name]
            else:
                optim_args['lr'] = self.lr
            self.optimizers[network_name] = optim_cls(network.parameters(), **optim_args)

    def _create_scheduler(self, opt):
        sched_args = dict()
        ext_args = defaultdict(dict)
        if opt.scheduler == 'step':
            sched_cls = optim.lr_scheduler.StepLR
            step_cnt = 4
            sched_args['step_size'] = opt.num_epochs // step_cnt
            sched_args['gamma'] = opt.lr_decay ** (1 / step_cnt)
        elif opt.scheduler == 'exp':
            sched_cls = optim.lr_scheduler.ExponentialLR
            sched_args['gamma'] = opt.lr_decay ** (1 / opt.num_epochs)
        elif opt.scheduler == 'cos':
            sched_cls = optim.lr_scheduler.CosineAnnealingLR
            # sched_args['eta_min'] = self.lr * opt.lr_decay
            sched_args['T_max'] = opt.num_epochs
        else:
            raise NameError(f'scheduler named {opt.scheduler} not defined')
        if opt.scheduler == 'cos':
            for network_name, network in self.model.networks.items():
                if isinstance(self.lr, dict):
                    ext_args['eta_min'][network_name] = self.lr[network_name] * opt.lr_decay
                else:
                    ext_args['eta_min'][network_name] = self.lr * opt.lr_decay
        self.schedulers = dict()
        for model_name, optimizer in self.optimizers.items():
            for key, value in ext_args.items():
                sched_args[key] = value[model_name]
            self.schedulers[model_name] = sched_cls(optimizer, **sched_args)
        # self.schedulers = {
        #     model_name: sched_cls(optimizer, **sched_args)
        #     for model_name, optimizer in self.optimizers.items()
        # }
        # initial lr
        for _ in range(self.first_epoch):
            for scheduler in self.schedulers.values():
                scheduler.step()

    def _update_per_epoch(self, epoch=None):
        """called after each training epoch"""
        for model_name in self.schedulers.keys():
            self.schedulers[model_name].step()
