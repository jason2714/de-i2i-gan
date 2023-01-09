from options.base_options import BaseOptions


class BaseTestOptions:
    def __init__(self):
        self.is_train = False

    def initialize(self, parser):
        parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.set_defaults(phase='test')
        return parser
