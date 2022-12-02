from options.base_options import BaseOptions


class BaseTestOptions:
    def __init__(self):
        self.isTrain = False

    def initialize(self, parser):
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.set_defaults(phase='test')
        return parser
