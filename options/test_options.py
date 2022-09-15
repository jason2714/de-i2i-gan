from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()
        self.isTrain = False

    def initialize(self, parser):
        parser = super(TestOptions, self).initialize(parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.set_defaults(phase='test')
        return parser
