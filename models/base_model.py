import torch.nn
from models.networks import save_network, load_network
import inspect
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, l1_loss, mse_loss


class BaseModel:
    def __init__(self, opt):
        """
        model name must start with 'net'
        example: netG, netD
        Notice!!!
        all other attributes should not start with 'net'
        """
        self.opt = opt
        self.network_prefix = 'net'

    @property
    def networks(self):
        """
        networks dictionary with
        key: network_name
        value: network_instance
        """
        # attributes = inspect.getmembers(self, lambda attr: not (inspect.isroutine(attr)))
        # networks_dict = {
        #     attr[0].replace(network_prefix, ''): attr[1]
        #     for attr in attributes
        #     if attr[0].startswith(network_prefix) and not (attr[0].startswith('__') and attr[0].endswith('__'))
        # }
        networks = {
            attr_name.replace(self.network_prefix, ''): attr_value
            for attr_name, attr_value in self.__dict__.items()
            if attr_name.startswith(self.network_prefix) and isinstance(attr_value, torch.nn.Module)
        }
        return networks

    def init_weights(self):
        """
            skip network initialized with name ends with _
        """
        print(f"initialize model's parameters using {self.opt.init_type} with variance={self.opt.init_variance}")
        for network_name, network in self.networks.items():
            if not network_name.endswith('_'):
                network.init_weights(self.opt.init_type, self.opt.init_variance)

    def save(self, epoch):
        # print(f"save model's weights with epoch {epoch}")
        for network_name, network in self.networks.items():
            save_network(network, network_name, epoch, self.opt)

    def load(self, epoch):
        print(f"load model's weights from epoch {epoch}")
        for network_name, network in self.networks.items():
            load_network(network, network_name, epoch, self.opt)

    def load_network(self, network_name, epoch):
        print(f"load net_{network_name}'s weights from epoch {epoch}")
        load_network(self.networks[network_name], network_name, epoch, self.opt)

    def __repr__(self):
        model_repr = ''
        for network_name, network in self.networks.items():
            split_line = '=' * 50 + f'{self.network_prefix + network_name:^8}' + '=' * 50 + '\n'
            model_repr += split_line + repr(network) + '\n' + split_line
        return model_repr

    def _cal_loss(self, logits, targets, loss_type):
        """Compute loss
            input type for cce and bce is unnormalized logits"""
        if loss_type in ('bce', 'bce_logits'):
            return binary_cross_entropy_with_logits(logits, targets)
        elif loss_type in ('cce', 'cce_logits'):
            return cross_entropy(logits, targets)
        elif loss_type == 'l1':
            return l1_loss(logits, targets)
        elif loss_type in ('l2', 'mse'):
            return mse_loss(logits, targets)
        else:
            raise ValueError(f"loss_type: {loss_type} is invalid")
