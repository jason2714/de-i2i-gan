import math
from collections import defaultdict

import torch
from torch import optim

from models import find_model_using_name


class BaseTrainer:
    """
    Trainer receives the options and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt, iters_per_epoch):
        self.opt = opt
        self.model = find_model_using_name(opt.model)(opt)
        self.model.init_weights()
        self.losses = defaultdict(list)
        self.dis_outputs = defaultdict(list)
        self.iters = 0

        # initial epochs and iters
        if self.opt.num_epochs == -1:
            self.opt.num_epochs = math.ceil(self.opt.num_iters / (iters_per_epoch + 1e-12))
        self.opt.num_iters = self.opt.num_epochs * iters_per_epoch

        # initial optimizer and scheduler
        self._create_optimizer(opt)
        self._create_scheduler(opt)

    def _create_optimizer(self, opt):
        assert isinstance(self.opt.lr, (int, float, dict)), 'type of lr should be scalar or dict'
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
            optim_args['betas'] = (0.5, 0.999)
        else:
            raise NameError(f'optimizer named {opt.optimizer} not defined')
        self.optimizers = {
            network_name: optim_cls(network.parameters(),
                                    lr=self.opt.lr[network_name] if isinstance(self.opt.lr, dict)
                                    else self.opt.lr, **optim_args)
            for network_name, network in self.model.networks.items()
        }

    def _create_scheduler(self, opt):
        sched_args = dict()
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
            sched_args['eta_min'] = opt.lr * opt.lr_decay
            sched_args['T_max'] = opt.num_epochs
        else:
            raise NameError(f'scheduler named {opt.scheduler} not defined')
        self.schedulers = {
            model_name: sched_cls(optimizer, **sched_args)
            for model_name, optimizer in self.optimizers.items()
        }
