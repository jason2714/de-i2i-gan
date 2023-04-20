from metrics.inception import InceptionV3
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
        parser.add_argument('--label_nc', type=int, default=6, help='# of label classes')
        parser.add_argument('--embed_nc', type=int, default=768, help='# of embedding classes, [768 | 1024]')

        # for generator
        # parser.add_argument('--netG', type=str, default='defectgan', help='selects model to use for netG (wgan)')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in last conv layer')
        parser.add_argument('--num_scales', type=int, default=2, help='# of gen scale layers')
        parser.add_argument('--num_res', type=int, default=6, help='# of gen resnet layers')
        parser.add_argument('--add_noise', action='store_true', default=False, help='whether to add noise in generator')
        parser.add_argument('--style_norm_block_type', type=str, default='spade', help='[spade | sean | adain]')

        # for discriminator
        # parser.add_argument('--netD', type=str, default='defectgan', help='selects model to use for netD (wgan)')
        parser.add_argument('--ndf', type=int, default=64, help='# of dis filters in first conv layer')
        parser.add_argument('--num_layers', type=int, default=5, help='# of dis encode layers')

        # for model
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02,
                            help='variance of the initialization distribution')
        parser.add_argument('--cycle_gan', action='store_true', help='Whether to use cycleGAN architecture')
        parser.add_argument('--skip_conn', action='store_true', help='Whether to use skip connection architecture')
        parser.add_argument('--use_spectral', action='store_true', default=False,
                            help='whether to use spectral norm in conv block')

        # for inception model
        parser.add_argument('--dims', type=int, default=2048,
                            choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                            help=('Dimensionality of Inception features to use. '
                                  'By default, uses pool3 features'))
        parser.add_argument('--num_imgs', type=int, default=5_000, help='use # images to calculate FID score')
        parser.add_argument('--npz_path', type=str, default=None,
                            help='Paths to .npz statistic files (required if cal_fid is True)')

        # for style embeddings
        parser.add_argument('--embed_path', type=Path, default=None, help='Path to embedding file')
        parser.add_argument('--num_embeds', type=int, default=3, help='Number of embeddings to use')
        parser.add_argument('--sean_alpha', type=float, default=None, help='Initial alpha value for SEAN block, '
                                 'if 1 then use embedding only, if 0 then use latent only')

        return parser


class TrainOptions(DefectGanBaseOptions, BaseTrainOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        DefectGanBaseOptions.__init__(self)
        BaseTrainOptions.__init__(self)

    def initialize(self, parser):
        parser = DefectGanBaseOptions.initialize(self, parser)
        parser = BaseTrainOptions.initialize(self, parser)

        # for displays
        parser.add_argument('--num_display_images', type=int, default=8,
                            help='# of display images')
        parser.add_argument('--save_img_freq', type=int, default=4,
                            help='frequency of saving generated images at the end of epochs')
        # for training
        parser.add_argument('--optimizer', type=str, default='adam',
                            help='type of optimizer [sgd|rmsprop|adam|adamw]')
        parser.add_argument('--num_iters', type=int, default=500_000, help='how many epochs for learning')
        parser.add_argument('--lr', type=float, nargs='+', default=[2e-4],
                            help='initial learning rate for optimizer, '
                                 'e.g. [lr] or [lr_d, lr_g]')
        parser.add_argument('--lr_decay', type=float, default=5e-3, help='learning rate decay for optimizer')
        parser.add_argument('--loss_weight', type=int, nargs='+', default=[2, 5, 5, 5, 1],
                            help='aggregation weight for each loss, [cls_d, cls_g, rec, sd_cyc, sd_con]')
        # parser.add_argument('--clf_loss_type', type=str, default='bce', help='loss type of classifier [bce|cce]')

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
        parser.add_argument('--metrics', type=str, nargs='+', default=None,
                            help='metrics to calculate [fid|is|lpips]')
        parser.add_argument('--cal_mfid', action='store_true', default=False,
                            help='whether to calculate mFID score or not')
        parser.add_argument('--save_img_grid', action='store_true', default=False,
                            help='whether to save generated partial labeled image grids or not')
        parser.add_argument('--save_img', action='store_true', default=False,
                            help='whether to save generated image grids or not')
        parser.add_argument('--save_stats', action='store_true', default=False,
                            help='whether to save generated image stats or not')
        parser.add_argument('--cal_clf', action='store_true', default=False,
                            help='whether to calculate classifier accuracy or not')
        parser.add_argument('--vis_style_embeds', type=str, default=None,
                            help='whether to visualize sean embeddings or not, type of [hidden|mean]')

        return parser


class PreTrainOptions(DefectGanBaseOptions, BaseTrainOptions):
    def __init__(self):
        super(PreTrainOptions, self).__init__()
        DefectGanBaseOptions.__init__(self)
        BaseTrainOptions.__init__(self)

    def initialize(self, parser):
        parser = DefectGanBaseOptions.initialize(self, parser)
        parser = BaseTrainOptions.initialize(self, parser)

        parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        # parser.add_argument('--max_device_batch_size', type=int, default=32)
        parser.add_argument('--save_latest_freq', type=int, default=300,
                            help='frequency of saving latest checkpoints at the end of iters')

        # for displays
        parser.add_argument('--num_display_images', type=int, default=4,
                            help='# of display images')
        parser.add_argument('--save_img_freq', type=int, default=1,
                            help='frequency of saving generated images at the end of epochs')
        # for training
        parser.add_argument('--mask_ratio', type=float, default=0.75, help='ratio of masked area')
        parser.add_argument('--optimizer', type=str, default='adamw',
                            help='type of optimizer [sgd|rmsprop|adam|adamw]')
        parser.add_argument('--num_epochs', type=int, default=200, help='how many epochs for learning')
        parser.add_argument('--lr', type=float, nargs='+', default=[1.5e-4],
                            help='initial learning rate for optimizer, '
                                 'e.g. [lr] or [lr_d, lr_g]')
        parser.add_argument('--scheduler', type=str, default='cos', help='type of scheduler [step|exp|cos]')
        parser.add_argument('--lr_decay', type=float, default=0.05, help='learning rate decay for optimizer')
        parser.add_argument('--loss_weight', type=int, nargs='+', default=[10, 3, 1],
                            help='aggregation weight for each loss, [rec, cls_d, cls_g]')
        parser.add_argument('--num_critics', type=int, default=1,
                            help='number of discriminator iterations per generator iterations.')
        parser.add_argument('--split_training', action='store_true', default=False,
                            help='Whether to train each network individually')

        # for MAE
        parser.add_argument('--patch_size', type=int, default=8, help='masked patch size, must be power of 2')
        # parser.add_argument('--cycle_gan', type=bool, default=True, help='Whether to use cycleGAN architecture')

        return parser
