from utils.util import use_gpu
from pathlib import Path
import argparse
import torch
import pickle


class BaseOptions:
    def __init__(self):
        self.initialized = False
        self.is_train = False

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--name', type=str, default='exp',
                            help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--model', type=str, required=True, help='which model to use')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--ckpt_dir', type=Path, default='./ckpt', help='models are saved here')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # input/output sizes
        parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        parser.add_argument('--image_size', type=int, default=128, help='input image size')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        parser.add_argument('--data_dir', type=Path, default='./data')
        parser.add_argument('--dataset_name', type=str, required=True, help='which dataset to use')
        parser.add_argument('--load_from_opt_file', action='store_true',
                            help='load the options from checkpoints and use that as default')

        # for model
        parser.add_argument('--init_type', type=str, default='xavier',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02,
                            help='variance of the initialization distribution')
        parser.add_argument('--use_spectral', action='store_true', help='whether to use spectral norm in conv block')

        # for generator
        # parser.add_argument('--netG', type=str, default='defectgan', help='selects model to use for netG (wgan)')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in last conv layer')

        # for discriminator
        # parser.add_argument('--netD', type=str, default='defectgan', help='selects model to use for netD (wgan)')
        parser.add_argument('--ndf', type=int, default=64, help='# of dis filters in first conv layer')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        # set conflict_handler to 'resolve' will enable arguments overriding
        if not self.initialized:
            parser = argparse.ArgumentParser(
                conflict_handler='resolve',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # check if experiment name is set
        if opt.name == parser.get_default('name'):
            name = f'{opt.model}_{opt.num_critics}_{opt.lr}'
            parser.set_defaults(name=name)
            parser.set_defaults(use_default_name=True)
        else:
            parser.set_defaults(use_default_name=False)
        # # modify model-related parser options
        # model_name = opt.model
        # model_option_setter = models.get_option_setter(model_name)
        # parser = model_option_setter(parser, self.is_train)
        #
        # opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt):
        expr_dir = opt.ckpt_dir / opt.name
        expr_dir.mkdir(parents=True, exist_ok=True)
        file_name = expr_dir / 'opt'
        return file_name

    def save_options(self, opt):
        file_path = self.option_file_path(opt)
        with file_path.with_suffix('.txt').open('w') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with file_path.with_suffix('.pkl').open('wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(k=new_val)
        return parser

    def load_options(self, opt):
        file_path = self.option_file_path(opt)
        new_opt = pickle.load(file_path.with_suffix('.pkl').open('rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()
        opt.is_train = self.is_train  # train or test

        self.print_options(opt)
        if opt.is_train:
            self.save_options(opt)
            # initial num_epochs
            assert opt.num_epochs != -1 or opt.num_iters == -1, \
                'Not define nums_epochs or num_iters in TrainOptions'

        # set gpu ids
        opt.device = use_gpu(opt.gpu_ids)

        # check if gpu-ids is correct
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        assert len(opt.gpu_ids) == 0 or opt.batch_size % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batchSize, len(opt.gpu_ids))
        # check if gpu-ids is correct

        self.opt = opt
        return self.opt
