import torch.nn
from models.networks import save_network, load_network
import inspect


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
        networks = {
            attr_name.replace(self.network_prefix, ''): attr_value
            for attr_name, attr_value in self.__dict__.items()
            if attr_name.startswith(self.network_prefix) and isinstance(attr_value, torch.nn.Module)
        }
        return networks

    def init_weights(self):
        for network in self.networks.values():
            network.init_weights(self.opt.init_type, self.opt.init_variance)

    def save(self, epoch):
        for network_name, network in self.networks.items():
            save_network(network, network_name, epoch, self.opt)

    def load(self, epoch):
        for network_name, network in self.networks.items():
            load_network(network, network_name, epoch, self.opt)

    def __repr__(self):
        model_repr = ''
        for network_name, network in self.networks.items():
            split_line = '=' * 50 + f'{self.network_prefix + network_name:^8}' + '=' * 50 + '\n'
            model_repr += split_line + repr(network) + '\n' + split_line
        return model_repr
