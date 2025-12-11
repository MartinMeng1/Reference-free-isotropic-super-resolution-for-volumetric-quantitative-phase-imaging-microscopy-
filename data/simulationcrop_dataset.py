# TODO SEP 08 version
import os.path
import numpy as np
from skimage import io

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from util import util


def _read_volume(path):
    # Supports .npy or typical image stacks like .tif/.tiff
    if str(path).lower().endswith('.npy'):
        return np.load(path)
    return io.imread(path)


class SimulationCropDataSet(BaseDataset):
    """
    Load ONE 3D volume and dice it into cubes (Z,Y,X order).
    Use with: --dataset_mode simulationcrop
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=False):
        parser.add_argument('--overlap', type=int, default=0,
                            help='overlap voxels between adjacent cubes')
        parser.add_argument('--border_cut', type=int, default=0,
                            help='crop this many voxels from each cube edge (removed after inference)')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        # exactly one file from dataroot
        self.A_path = make_dataset(opt.dataroot, 1)[0]
        self.roi_size = opt.dice_size[0]
        self.overlap = opt.overlap
        self.border_cut = opt.border_cut

        # optional manual crop window (keep or remove as you like)
        z0, y0, x0 = 136, 104, 120
        vol = _read_volume(self.A_path)
        # If too small for that crop, just keep whole volume
        if (vol.shape[0] >= z0 + 700) and (vol.shape[1] >= y0 + 700) and (vol.shape[2] >= x0 + 700):
            vol = vol[z0:z0+700, y0:y0+700, x0:x0+700]

        self.transform = get_transform(opt)

        # Remember original size to unpad in assembler
        self.image_size_original = vol.shape

        # Pad to tile cleanly
        vol = util.pad_for_dicing(vol, self.roi_size, overlap=self.overlap)
        self.image_size = vol.shape

        self.cube = DiceCube(vol, self.roi_size, overlap=self.overlap, border_cut=self.border_cut)

    def __getitem__(self, index):
        cube = self.cube[index]                 # numpy [Z,Y,X] (with extra border if border_cut>0)
        A = self.transform(cube)                # torch [C,D,H,W] with C=1
        return {'A': A, 'A_paths': str(index)}  # index is our synthetic path

    def __len__(self):
        return len(self.cube)

    def shape(self):
        return (self.cube.z_steps, self.cube.y_steps, self.cube.x_steps)

    def size(self):
        return self.image_size

    def size_original(self):
        return self.image_size_original


class DiceCube:
    """Dices a volume into overlapping cubes in x→y→z order."""
    def __init__(self, image, roi_size, overlap=0, border_cut=0):
        self.image = image
        self.roi_size = int(roi_size)
        self.overlap = int(overlap)
        self.border_cut = int(border_cut)

        self.step = self.roi_size - self.overlap
        assert self.step > 0, "dice step must be > 0 (roi_size - overlap)"

        self.z_steps = (self.image.shape[0] - self.overlap) // self.step
        self.y_steps = (self.image.shape[1] - self.overlap) // self.step
        self.x_steps = (self.image.shape[2] - self.overlap) // self.step

        # pad for safe border cropping later
        npad = ((border_cut, border_cut),
                (border_cut, border_cut),
                (border_cut, border_cut))
        if any(p > 0 for p in (border_cut,)):
            self.image = np.pad(self.image, pad_width=npad, mode='reflect')

    def indexToCoordinates(self, index):
        x_idx = index % self.x_steps
        y_idx = (index % (self.x_steps * self.y_steps)) // self.x_steps
        z_idx = index // (self.x_steps * self.y_steps)
        return z_idx, y_idx, x_idx

    def __getitem__(self, index):
        z_idx, y_idx, x_idx = self.indexToCoordinates(index)
        cz = z_idx * self.step + self.border_cut
        cy = y_idx * self.step + self.border_cut
        cx = x_idx * self.step + self.border_cut

        # include extra border for later cutting inside the model/assembler
        return self.image[
            cz - self.border_cut: cz + self.roi_size + self.border_cut,
            cy - self.border_cut: cy + self.roi_size + self.border_cut,
            cx - self.border_cut: cx + self.roi_size + self.border_cut,
        ]

    def __len__(self):
        return self.x_steps * self.y_steps * self.z_steps
