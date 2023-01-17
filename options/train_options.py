from options.base_options import BaseOptions
from pathlib import Path


class BaseTrainOptions:
    def __init__(self):
        self.is_train = True

    def initialize(self, parser):
        # for displays
        parser.add_argument('--num_display_images', type=int, default=64,
                            help='# of display images')
        parser.add_argument('--save_epoch_freq', type=int, default=10,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_latest_freq', type=int, default=1000,
                            help='frequency of saving latest checkpoints at the end of iters')

        # for lr
        parser.add_argument('--optimizer', type=str, required=True, help='type of optimizer [sgd|rmsprop|adam|adamw]')
        parser.add_argument('--lr', type=float, required=True, help='initial learning rate for optimizer')
        parser.add_argument('--num_epochs', type=int, default=-1, help='how many epochs for training')
        parser.add_argument('--num_iters', type=int, default=-1, help='how many iters for training, '
                                                                      'ignored when num_epochs defined!!')

        # for lr_decay
        parser.add_argument('--scheduler', type=str, default='step', help='type of scheduler [step|exp|cos]')
        parser.add_argument('--lr_decay', type=float, default=1, help='learning rate decay for optimizer')
        # parser.add_argument('--niter_decay', type=int, default=0,
        #                     help='# of iter to linearly decay learning rate to zero')

        parser.add_argument('--num_critics', type=int, default=1,
                            help='number of discriminator iterations per generator iterations.')
        parser.add_argument('--continue_training', action='store_true', help='continue training: load the latest model')

        # for logging
        parser.add_argument('--log_dir', type=Path, default='./log', help='directory of tensorboard log')
        # parser.add_argument('--log_comment', type=str, required=True, help='comment of tensorboard log')

        return parser

    def init_num_iters(self):
        """
            transform num_iters from num_epochs
        """
        pass
