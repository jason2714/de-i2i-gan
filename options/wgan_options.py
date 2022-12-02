from options.base_options import BaseOptions
from options.train_options import BaseTrainOptions
from options.test_options import BaseTestOptions
from pathlib import Path


class WGanBaseOptions(BaseOptions):
    def __init__(self):
        super(WGanBaseOptions, self).__init__()

    def initialize(self, parser):
        # overriding base options
        parser = super(WGanBaseOptions, self).initialize(parser)

        parser.add_argument('--name', type=str, default='exp',
                            help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--model', type=str, default='wgan', help='which model to use')

        # for setting input/output
        parser.add_argument('--dataset_name', type=str, default='face', help='which dataset to use')
        parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
        parser.add_argument('--image_size', type=int, default=64, help='input image size')

        # for generator
        # parser.add_argument('--netG', type=str, default='wgan', help='selects model to use for netG (wgan)')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in last conv layer')

        # for discriminator
        # parser.add_argument('--netD', type=str, default='wgan', help='selects model to use for netD (wgan)')
        parser.add_argument('--ndf', type=int, default=64, help='# of dis filters in first conv layer')

        # for model specific arguments
        parser.add_argument('--noise_dim', type=int, default=100, help="dimension of the latent z vector")
        parser.add_argument('--clipping_limit', type=float, default=0.03, help='clipping limit of W-GAN')

        return parser


class TrainOptions(WGanBaseOptions, BaseTrainOptions):
    def __init__(self):
        WGanBaseOptions.__init__(self)
        BaseTrainOptions.__init__(self)

    def initialize(self, parser):
        parser = WGanBaseOptions.initialize(self, parser)
        parser = BaseTrainOptions.initialize(self, parser)

        # for training
        parser.add_argument('--optimizer', type=str, default='rmsprop',
                            help='type of optimizer [sgd|rmsprop|adam|adamw]')
        parser.add_argument('--num_epochs', type=int, default=120, help='how many epochs for learning')
        parser.add_argument('--lr', type=float, default=5e-5, help='initial learning rate for optimizer')
        # parser.add_argument('--niter_decay', type=int, default=0,
        #                     help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--num_critics', type=int, default=5,
                            help='number of discriminator iterations per generator iterations.')

        return parser


class TestOptions(WGanBaseOptions, BaseTestOptions):
    def __init__(self):
        WGanBaseOptions.__init__(self)
        BaseTestOptions.__init__(self)

    def initialize(self, parser):
        WGanBaseOptions.initialize(self, parser)
        BaseTestOptions.initialize(self, parser)
        # for testing
        return parser
