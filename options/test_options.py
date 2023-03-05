from options.base_options import BaseOptions
from pathlib import Path


class BaseTestOptions:
    def __init__(self):
        self.is_train = False

    def initialize(self, parser):
        parser.add_argument('--results_dir', type=Path, default='./results/', help='saves results here.')
        parser.set_defaults(phase='test')
        return parser
