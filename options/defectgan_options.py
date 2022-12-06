from options.base_options import BaseOptions
from options.train_options import BaseTrainOptions
from options.test_options import BaseTestOptions
from pathlib import Path


class DefectGanBaseOptions(BaseOptions):
    def __init__(self):
        super(DefectGanBaseOptions, self).__init__()

    def initialize(self, parser):
        # overriding base options
        parser = super(DefectGanBaseOptions, self).initialize(parser)

        parser.add_argument('--name', type=str, default='exp',
                            help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--model', type=str, default='defectgan', help='which model to use')

        # for setting input/output
        parser.add_argument('--dataset_name', type=str, default='codebrim', help='which dataset to use')
        parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        parser.add_argument('--image_size', type=int, default=128, help='input image size')

        # for generator
        # parser.add_argument('--netG', type=str, default='defectgan', help='selects model to use for netG (wgan)')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in last conv layer')

        # for discriminator
        # parser.add_argument('--netD', type=str, default='defectgan', help='selects model to use for netD (wgan)')
        parser.add_argument('--ndf', type=int, default=64, help='# of dis filters in first conv layer')

        return parser


class TrainOptions(DefectGanBaseOptions, BaseTrainOptions):
    def __init__(self):
        DefectGanBaseOptions.__init__(self)
        BaseTrainOptions.__init__(self)

    def initialize(self, parser):
        parser = DefectGanBaseOptions.initialize(self, parser)
        parser = BaseTrainOptions.initialize(self, parser)

        # for training
        parser.add_argument('--optimizer', type=str, default='adam',
                            help='type of optimizer [sgd|rmsprop|adam|adamw]')
        parser.add_argument('--num_iters', type=int, default=500_000, help='how many epochs for learning')
        parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate for optimizer')
        parser.add_argument('--lr_decay', type=float, default=5e-3, help='learning rate decay for optimizer')

        # parser.add_argument('--niter_decay', type=int, default=0,
        #                     help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--num_critics', type=int, default=5,
                            help='number of discriminator iterations per generator iterations.')

        return parser


class TestOptions(DefectGanBaseOptions, BaseTestOptions):
    def __init__(self):
        DefectGanBaseOptions.__init__(self)
        BaseTestOptions.__init__(self)

    def initialize(self, parser):
        DefectGanBaseOptions.initialize(self, parser)
        BaseTestOptions.initialize(self, parser)
        # for testing
        return parser
