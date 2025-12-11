# TODO SEP 08 version
"""A modified image/volume folder helper.

- Recursively collects files with supported extensions.
- Loader supports:
  * 2D images via PIL (jpg/png/bmp/...)
  * 3D stacks via skimage.io.imread (tif/tiff)
  * NumPy volumes via np.load (.npy)
"""

import os
import os.path
import re
from typing import List

import numpy as np
from PIL import Image
from skimage import io as skio
import torch.utils.data as data

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp',
    '.tif', '.tiff',
    '.npy'
)

def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMG_EXTENSIONS)

def make_dataset(dir: str, max_dataset_size=float("inf")) -> List[str]:
    assert os.path.isdir(dir), f'{dir} is not a valid directory'
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if not fname.startswith('.') and is_image_file(fname):
                images.append(os.path.join(root, fname))
    # keep order stable
    return images[:min(int(max_dataset_size), len(images))]

def merge_datasets(dirs, max_dataset_size=float("inf")):
    image_set = []
    for d in dirs:
        image_set += make_dataset(d, max_dataset_size)
    return image_set

def default_loader(path: str):
    plower = path.lower()
    if plower.endswith('.npy'):
        return np.load(path)                  # expects (Z,Y,X) or (H,W) or (Z,Y,X,C)
    if plower.endswith('.tif') or plower.endswith('.tiff'):
        return skio.imread(path)              # supports 2D or 3D stacks
    # fall back to PIL for standard 2D images
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, return_paths=False, loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise RuntimeError(
                "Found 0 images in: " + root + "\n"
                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
            )
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        return img

    def __len__(self):
        return len(self.imgs)
