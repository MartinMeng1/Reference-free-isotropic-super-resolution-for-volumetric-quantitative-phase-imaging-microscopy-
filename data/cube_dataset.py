import os.path
import re
import numpy as np
from skimage import io

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def _read_volume(path: str):
    """Read a 3D volume. Supports .npy or typical image stacks."""
    p = str(path).lower()
    if p.endswith('.npy'):
        return np.load(path)             # expects (Z,Y,X) or (Z,Y,X,C)
    return io.imread(path)               # e.g., .tif/.tiff stack → (Z,Y,X)


class CubeDataset(BaseDataset):
    """
    Loads a dataset of multiple 3D volumes from a directory.
    Each item is one whole volume (before transforms).
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        # collect file paths (recursively)
        self.A_paths = make_dataset(opt.dataroot)
        self.A_paths.sort(key=numericalSort)
        self.A_size = len(self.A_paths)
        if self.A_size == 0:
            raise RuntimeError(f"No volumes found in: {opt.dataroot}")

        self.transform_A = get_transform(self.opt)
        self.isTrain = opt.isTrain

    def __getitem__(self, index):
        # keep index in range
        A_path = self.A_paths[index % self.A_size]

        # read volume (Z,Y,X)
        A_vol = _read_volume(A_path)

        # apply transform → torch float tensor [C,D,H,W]
        A = self.transform_A(A_vol)

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        return self.A_size
