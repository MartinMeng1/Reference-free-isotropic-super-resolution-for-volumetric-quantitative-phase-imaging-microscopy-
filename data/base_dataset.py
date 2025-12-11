# data/base_dataset.py  — robust 2D/3D base dataset & transforms (SEP 08)
import random
import math
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
from scipy.ndimage import rotate as scipy_rotate


class BaseDataset(data.Dataset, ABC):
    """
    Abstract base class for datasets.

    Subclasses must implement:
      - __init__(self, opt)
      - __len__(self)
      - __getitem__(self, index)
      - (optional) modify_commandline_options(parser, is_train)
    """
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass


# -----------------------------
# Transform helpers
# -----------------------------
def get_params(opt, vol_shape):
    """
    Produce deterministic transform params for a single crop/flip/rot.
    vol_shape: (Z, Y, X) for 3D or (H, W) for 2D (but we use 3D here).
    """
    crop_z, crop_y, crop_x = tuple(opt.crop_size)
    assert vol_shape[0] >= crop_z and vol_shape[1] >= crop_y and vol_shape[2] >= crop_x, \
        f"Volume {vol_shape} smaller than crop {opt.crop_size}"

    z = random.randint(0, max(0, vol_shape[0] - crop_z))
    y = random.randint(0, max(0, vol_shape[1] - crop_y))
    x = random.randint(0, max(0, vol_shape[2] - crop_x))
    flip_axis = np.random.randint(0, 3)  # 0=Z,1=Y,2=X
    angle_3D = random.randint(0, 359)

    return {"crop_pos": (z, y, x), "flip_axis": int(flip_axis), "angle_3D": int(angle_3D)}


def get_transform(opt, params=None):
    """
    Compose transforms according to opt.preprocess (list of tokens).
    Order is kept close to your original:
      random3Drotate -> random90rotate -> random/center crop -> normalize -> flip -> channels -> toTensor
    """
    tfm = []
    image_dimension = int(getattr(opt, "image_dimension", 3))  # not used directly, kept for compatibility

    if "random3Drotate" in opt.preprocess:
        if params is None:
            tfm += [T.Lambda(lambda v: _random_rotate_clean_3d_xy(v))]
        else:
            tfm += [T.Lambda(lambda v: _rotate_clean_3d_xy(v, angle=params["angle_3D"]))]

    if "random90rotate" in opt.preprocess:
        if params is None:
            tfm += [T.Lambda(lambda v: _random90_rotate(v))]
        else:
            # if you want deterministic 90° rotations, pass ('angle','axis') via params['rotate_params']
            tfm += [T.Lambda(lambda v: _rotate_axes(v, params.get("rotate_params", (0, (1, 2)))))]  # default no-op

    if "randomcrop" in opt.preprocess:
        if params is None:
            tfm += [T.Lambda(lambda v: _random_crop(v, tuple(opt.crop_size)))]
        else:
            tfm += [T.Lambda(lambda v: _crop_at(v, params["crop_pos"], tuple(opt.crop_size)))]

    if "centercrop" in opt.preprocess:
        tfm += [T.Lambda(lambda v: _center_crop(v, opt.crop_portion))]

    tfm += [T.Lambda(lambda v: _normalize_01(v))]

    if "randomflip" in opt.preprocess:
        if params is None:
            tfm += [T.Lambda(lambda v: _random_flip(v))]
        else:
            tfm += [T.Lambda(lambda v: _flip_axis(v, params["flip_axis"]))]

    if "addColorChannel" in opt.preprocess:
        tfm += [T.Lambda(lambda v: _add_color_channel(v))]

    if "reorderColorChannel" in opt.preprocess:
        tfm += [T.Lambda(lambda v: _reorder_color_cyx(v))]

    if "addBatchChannel" in opt.preprocess:
        # keep behavior identical to your previous code (alias to addColorChannel)
        tfm += [T.Lambda(lambda v: _add_color_channel(v))]

    tfm += [T.Lambda(lambda v: _to_tensor_channels_first(v))]
    return T.Compose(tfm)


# -----------------------------
# Primitive ops (robust versions)
# -----------------------------
def _normalize_01(arr: np.ndarray) -> np.ndarray:
    """Normalize to [0,1]. Integers: scale by dtype max. Floats: min-max per array."""
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        out = arr.astype(np.float32) / float(info.max)
    elif np.issubdtype(arr.dtype, np.floating):
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if vmax > vmin:
            out = ((arr - vmin) / (vmax - vmin)).astype(np.float32)
        else:
            out = np.zeros_like(arr, dtype=np.float32)
    else:
        raise TypeError(f"Unsupported dtype: {arr.dtype}")
    return out


def _random90_rotate(vol: np.ndarray) -> np.ndarray:
    """Rotate each slice in Z by a random k*90 degrees (keeps spatial size)."""
    k = np.random.choice([1, 2, 3])  # 90, 180, 270
    # rotate in-plane over (Y,X) == axes (1,2)
    return np.rot90(vol, k=k, axes=(1, 2)).copy()


def _rotate_axes(arr: np.ndarray, rotate_params):
    """General 3D rotate with scipy (deterministic if provided)."""
    angle, axis = rotate_params
    return scipy_rotate(arr, angle=angle, axes=axis, reshape=False, mode="reflect")


def _random_crop(arr: np.ndarray, crop_size):
    """Handles 2D and 3D arrays safely."""
    if arr.ndim == 3:
        cz, cy, cx = crop_size
        Z, Y, X = arr.shape
        assert Z >= cz and Y >= cy and X >= cx
        z = np.random.randint(0, Z - cz + 1) if cz > 0 else 0
        y = np.random.randint(0, Y - cy + 1) if cy > 0 else 0
        x = np.random.randint(0, X - cx + 1) if cx > 0 else 0
        z2 = z + cz if cz > 0 else None
        y2 = y + cy if cy > 0 else None
        x2 = x + cx if cx > 0 else None
        return arr[z:z2, y:y2, x:x2].copy()
    elif arr.ndim == 2:
        cy, cx = crop_size[1], crop_size[2] if len(crop_size) == 3 else crop_size
        H, W = arr.shape
        assert H >= cy and W >= cx
        y = np.random.randint(0, H - cy + 1) if cy > 0 else 0
        x = np.random.randint(0, W - cx + 1) if cx > 0 else 0
        y2 = y + cy if cy > 0 else None
        x2 = x + cx if cx > 0 else None
        return arr[y:y2, x:x2].copy()
    else:
        raise ValueError(f"Expected 2D/3D array, got shape {arr.shape}")


def _crop_at(arr: np.ndarray, pos, crop_size):
    if arr.ndim == 3:
        z, y, x = pos
        cz, cy, cx = crop_size
        return arr[z:z + cz, y:y + cy, x:x + cx].copy()
    elif arr.ndim == 2:
        y, x = pos[1], pos[2]
        cy, cx = crop_size[1], crop_size[2]
        return arr[y:y + cy, x:x + cx].copy()
    else:
        raise ValueError(f"Expected 2D/3D array, got shape {arr.shape}")


def _center_crop(arr: np.ndarray, crop_portion: float):
    """
    crop_portion is given as percentage kept (like your original intent):
      if opt.crop_portion = 90, we crop 10% away => keep 90%.
    """
    keep = float(crop_portion) / 100.0
    keep = np.clip(keep, 0.0, 1.0)
    if arr.ndim == 3:
        Z, Y, X = arr.shape
        cz, cy, cx = int(Z * keep), int(Y * keep), int(X * keep)
        sz, sy, sx = (Z - cz) // 2, (Y - cy) // 2, (X - cx) // 2
        return arr[sz:sz + cz, sy:sy + cy, sx:sx + cx].copy()
    elif arr.ndim == 2:
        H, W = arr.shape
        ch, cw = int(H * keep), int(W * keep)
        sy, sx = (H - ch) // 2, (W - cw) // 2
        return arr[sy:sy + ch, sx:sx + cw].copy()
    else:
        raise ValueError(f"Expected 2D/3D array, got shape {arr.shape}")


def _flip_axis(arr: np.ndarray, axis: int):
    return np.flip(arr, axis=axis).copy()


def _random_flip(arr: np.ndarray):
    out = arr
    for ax in range(arr.ndim):
        if np.random.rand() < 0.5:
            out = np.flip(out, axis=ax)
    return out.copy()


def _to_tensor_channels_first(arr: np.ndarray) -> torch.Tensor:
    """
    Ensure float32 contiguous & channels-first:
      - 3D grayscale volume Z,Y,X  -> add C=1 -> (1,Z,Y,X)
      - 2D grayscale image  Y,X    -> add C=1 -> (1,Y,X)
      - If already has channel dim first, leave as-is.
    """
    a = arr
    if a.ndim == 3:   # Z,Y,X (no channel)
        a = np.expand_dims(a, axis=0)  # -> C,Z,Y,X
    elif a.ndim == 2: # Y,X
        a = np.expand_dims(a, axis=0)  # -> C,Y,X
    # if user already added color channel earlier (C, Z, Y, X) or (C, Y, X), leave it
    a = np.ascontiguousarray(a, dtype=np.float32)
    return torch.from_numpy(a)


def _add_color_channel(arr: np.ndarray) -> np.ndarray:
    """Add leading channel dimension for grayscale arrays."""
    if arr.ndim == 3:      # Z,Y,X -> 1,Z,Y,X
        return np.expand_dims(arr, axis=0)
    elif arr.ndim == 2:    # Y,X -> 1,Y,X
        return np.expand_dims(arr, axis=0)
    else:
        return arr  # already has channel dim


def _reorder_color_cyx(arr: np.ndarray) -> np.ndarray:
    """Reorder from (Y,X,C) -> (C,Y,X). If 3D, assume (Z,Y,X) no-op."""
    if arr.ndim == 3 and arr.shape[-1] <= 4:  # (Y,X,C)
        return np.transpose(arr, (2, 0, 1)).copy()
    return arr


# ---- clean rotations (slice-wise OpenCV with crop-to-original-size) ----
def _rotate_clean(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 2D image around center and crop to original size."""
    h, w = img.shape[:2]
    img_rot = _rotate_image_full(img, angle)
    cw, ch = _largest_rotated_rect(w, h, math.radians(angle))
    cw, ch = int(round(cw)), int(round(ch))
    cropped = _crop_center(img_rot, cw, ch)
    # Pad back to original if rounding made it smaller
    return _pad_or_crop_to(cropped, (h, w))


def _rotate_clean_3d_xy(vol: np.ndarray, angle: float) -> np.ndarray:
    slices = [ _rotate_clean(sl, angle) for sl in vol ]
    return np.stack(slices, axis=0)


def _random_rotate_clean_3d_xy(vol: np.ndarray) -> np.ndarray:
    angle = random.randint(0, 359)
    return _rotate_clean_3d_xy(vol, angle)


# --- Utility funcs used by _rotate_clean ---
def _rotate_image_full(image, angle):
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # compute new bounds
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust translation
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR)

def _largest_rotated_rect(w, h, angle_rad):
    quadrant = int(math.floor(angle_rad / (math.pi / 2))) & 3
    sign_alpha = angle_rad if ((quadrant & 1) == 0) else math.pi - angle_rad
    alpha = (sign_alpha % math.pi + math.pi) % math.pi
    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)
    gamma = math.atan2(bb_w, bb_w)
    delta = math.pi - alpha - gamma
    length = min(w, h)
    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)
    y = a * math.cos(gamma)
    x = y * math.tan(gamma)
    return (bb_w - 2 * x, bb_h - 2 * y)

def _crop_center(image, width, height):
    H, W = image.shape[:2]
    width = min(W, width); height = min(H, height)
    cy, cx = H // 2, W // 2
    y1 = max(0, cy - height // 2); y2 = y1 + height
    x1 = max(0, cx - width // 2);  x2 = x1 + width
    return image[y1:y2, x1:x2]

def _pad_or_crop_to(img, shape_hw):
    Ht, Wt = shape_hw
    H, W = img.shape[:2]
    dy = max(0, Ht - H); dx = max(0, Wt - W)
    if dy > 0 or dx > 0:
        pad = ((dy // 2, dy - dy // 2), (dx // 2, dx - dx // 2))
        return np.pad(img, pad, mode="reflect")
    # crop if larger
    return img[:Ht, :Wt]
