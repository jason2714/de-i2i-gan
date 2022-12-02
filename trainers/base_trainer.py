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

    def __init__(self, opt):
        self.opt = opt
        self.model = find_model_using_name(opt.model)(opt)
        self.model.init_weights()
        self._create_optimizer(opt)
        self.losses = defaultdict(list)
        self.steps = 0

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
