from options.base_options import BaseOptions
from options.test_options import BaseTestOptions
from options.train_options import BaseTrainOptions


class ViTBaseOptions(BaseOptions):
    def __init__(self):
        super(ViTBaseOptions, self).__init__()

    def initialize(self, parser):
        # overriding base options
        parser = super(ViTBaseOptions, self).initialize(parser)

        parser.add_argument('--name', type=str, default='exp',
                            help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--model', type=str, default='vit', help='which model to use')

        # for setting input/output
        parser.add_argument('--dataset_name', type=str, default='codebrim', help='which dataset to use')
        parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
        parser.add_argument('--image_size', type=int, default=224, help='input image size')
        parser.add_argument('--label_nc', type=int, default=6, help='# of label classes')

        # for model
        parser.add_argument('--model_size', type=str, default='base', help='model size [base|large]')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02,
                            help='variance of the initialization distribution')

        return parser


class TrainOptions(ViTBaseOptions, BaseTrainOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        ViTBaseOptions.__init__(self)
        BaseTrainOptions.__init__(self)

    def initialize(self, parser):
        parser = ViTBaseOptions.initialize(self, parser)
        parser = BaseTrainOptions.initialize(self, parser)

        # for training
        parser.add_argument('--optimizer', type=str, default='adamw',
                            help='type of optimizer [sgd|rmsprop|adam|adamw]')
        parser.add_argument('--scheduler', type=str, default='cos', help='type of scheduler [step|exp|cos]')
        parser.add_argument('--num_epochs', type=int, default=50, help='how many epochs for learning')
        parser.add_argument('--lr', type=float, nargs='+', default=[5e-4],
                            help='initial learning rate for optimizer, '
                                 'e.g. [lr] or [lr_d, lr_g]')
        parser.add_argument('--lr_decay', type=float, default=2e-4, help='learning rate decay for optimizer')

        return parser


class TestOptions(ViTBaseOptions, BaseTestOptions):
    def __init__(self):
        ViTBaseOptions.__init__(self)
        BaseTestOptions.__init__(self)

    def initialize(self, parser):
        ViTBaseOptions.initialize(self, parser)
        BaseTestOptions.initialize(self, parser)

        # for testing
        parser.add_argument('--save_embeddings', action='store_true', default=False,
                            help='whether to save the embedding of the test images')
        parser.add_argument('--visualize_tsne', action='store_true', default=False,
                            help='whether to visualize tsne')
        parser.add_argument('--calc_classifier_acc', action='store_true', default=False,
                            help='whether to calculate classifier accuracy')
        parser.add_argument('--data_type', type=str, default='fusion',
                            help='which datatype to use, [ defects | background | fusion ]')

        return parser
