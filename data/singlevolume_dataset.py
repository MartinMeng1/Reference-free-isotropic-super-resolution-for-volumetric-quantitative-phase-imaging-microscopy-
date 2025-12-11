# TODO SEP 08 version
import numpy as np
from skimage import io

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset


def _read_volume(path):
    if str(path).lower().endswith('.npy'):
        return np.load(path)
    return io.imread(path)


class SingleVolumeDataset(BaseDataset):
    """
    Loads a single 3D volume and returns the whole thing each __getitem__.
    Use with: --dataset_mode single_volume
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.A_path = make_dataset(opt.dataroot, 1)[0]
        self.A_img_np = _read_volume(self.A_path)   # (Z,Y,X) or (Z,Y,X,C)

        self.transform_A = get_transform(self.opt)
        self.isTrain = opt.isTrain

        # how many iterations per epoch to simulate
        self._epoch_len = getattr(opt, "epoch_len", 10)

    @staticmethod
    def modify_commandline_options(parser, is_train=False):
        parser.add_argument('--epoch_len', type=int, default=10,
                            help='virtual length (batches) per epoch for SingleVolumeDataset')
        return parser

    def __getitem__(self, index):
        A = self.transform_A(self.A_img_np)     # -> torch [C,D,H,W]
        return {'A': A, 'A_paths': self.A_path}

    def __len__(self):
        return int(self._epoch_len)