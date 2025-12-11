# util/util.py
# TODO Nov 01 hardened + 3D-safe version
"""This module contains simple helper functions."""

from __future__ import print_function
import os
import math
from typing import Tuple, Union

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# -----------------------------
# Image/Tensor <-> NumPy helpers
# -----------------------------

def tensor2im(input_image: Union[torch.Tensor, np.ndarray],
              imtype=np.uint16) -> np.ndarray:
    """
    Convert a Tensor/ndarray in [0,1] (or arbitrary float range) to a NumPy image array
    with the requested dtype (default uint16). Handles 2D/3D gracefully by ignoring
    non-spatial axes.

    Args:
        input_image: torch.Tensor or np.ndarray. If Tensor, any device/dtype OK.
        imtype: np.uint8 | np.uint16 | float

    Returns:
        np.ndarray with dtype imtype
    """
    if isinstance(input_image, torch.Tensor):
        image_numpy = input_image.detach().cpu().float().numpy()
    elif isinstance(input_image, np.ndarray):
        image_numpy = input_image.copy()
    else:
        return input_image  # passthrough

    # Bring to [0,1] for integer outputs; leave as float for np.float32
    if imtype in (np.uint8, np.uint16):
        # Robust min-max per array to avoid clipping if range not [0,1]
        vmin = float(np.nanmin(image_numpy))
        vmax = float(np.nanmax(image_numpy))
        if vmax > vmin:
            image_numpy = (image_numpy - vmin) / (vmax - vmin)
        else:
            image_numpy = np.zeros_like(image_numpy, dtype=np.float32)

    if imtype == np.uint8:
        image_numpy = np.clip(image_numpy, 0.0, 1.0) * 255.0
        image_numpy = np.clip(image_numpy, 0, 255)
    elif imtype == np.uint16:
        image_numpy = np.clip(image_numpy, 0.0, 1.0) * (2**16 - 1)
        image_numpy = np.clip(image_numpy, 0, 2**16 - 1)
    elif imtype in (np.float32, float):
        image_numpy = image_numpy.astype(np.float32)
        return image_numpy

    return image_numpy.astype(imtype)


# -----------------------------
# Normalization / Metrics
# -----------------------------

def normalize(img_np: np.ndarray, data_type= float) -> np.ndarray:
    """
    Min-max normalize to requested dtype range.
    data_type may be np.uint8, np.uint16, np.float32/float.
    """
    img_np = np.asarray(img_np)
    vmin = float(np.min(img_np))
    vmax = float(np.max(img_np))
    if vmax <= vmin:
        if data_type in (np.float32, float, np.float64):
            return np.zeros_like(img_np, dtype=np.float32)
        rng_max = 255 if data_type == np.uint8 else (2**16 - 1)
        return np.zeros_like(img_np, dtype=np.uint16 if rng_max > 255 else np.uint8)

    if data_type == np.uint8:
        new_max = 255.0
        out = (img_np - vmin) * (new_max / (vmax - vmin))
        return out.astype(np.uint8)
    elif data_type == np.uint16:
        new_max = float(2**16 - 1)
        out = (img_np - vmin) * (new_max / (vmax - vmin))
        return out.astype(np.uint16)
    else:  # float
        out = (img_np - vmin) / (vmax - vmin)
        return out.astype(np.float32)


def get_mse(source: np.ndarray, target: np.ndarray) -> float:
    source = source.astype(np.float64)
    target = target.astype(np.float64)
    return float(np.mean((target - source) ** 2))


def get_psnr(source: np.ndarray, target: np.ndarray, data_range: float) -> float:
    """
    PSNR in dB with numerical safety.
    data_range: max - min of the signal range (e.g., 1.0 for [0,1], 255 for uint8, 65535 for uint16)
    """
    source = source.astype(np.float64)
    target = target.astype(np.float64)
    mse = np.mean((target - source) ** 2)
    eps = 1e-12
    mse = max(mse, eps)
    return 20.0 * math.log10(float(data_range)) - 10.0 * math.log10(mse)


def get_snr(img_original: np.ndarray, img_noised: np.ndarray) -> float:
    """
    SNR (signal power / noise power) in dB.
    """
    x = img_original.astype(np.float64)
    y = img_noised.astype(np.float64)
    noise = y - x
    ps = float(np.mean(x ** 2))
    pn = float(np.mean(noise ** 2))
    eps = 1e-12
    return 10.0 * math.log10(max(ps, eps) / max(pn, eps))


def standardize(img_np: np.ndarray) -> np.ndarray:
    img_np = img_np.astype(np.float32)
    mu = float(np.mean(img_np))
    sigma = float(np.std(img_np))
    if sigma <= 0:
        return np.zeros_like(img_np, dtype=np.float32)
    return (img_np - mu) / sigma


# -----------------------------
# Noise models
# -----------------------------

def noisy(noise_typ: str,
          image: Union[np.ndarray, torch.Tensor],
          sigma: float = 0.1,
          peak: float = 0.1,
          is_tensor: bool = False,
          is_normalize: bool = True):
    """
    Add Gaussian or Poisson noise.

    Args:
        noise_typ: "gauss" or "poisson"
        image: np.ndarray or torch.Tensor. If Tensor, expects shape (B,C,Z,Y,X) or (B,C,H,W) or (C,...) etc.
        sigma: std-dev for Gaussian noise (same units as image)
        peak: Poisson peak scale; output = Poisson(image * peak) / peak
        is_tensor: if True, return a Tensor on same device/dtype as input tensor; otherwise NumPy
        is_normalize: if True, min-max normalize result to [0,1] (float32)

    Returns:
        Noisy image with same layout; dtype float32 for numpy or float tensor for torch.
    """
    if isinstance(image, torch.Tensor):
        img_t = image.detach()
        device = img_t.device
        out_tensor = True
        img = img_t.float()
        img_np = img.cpu().numpy()
    else:
        device = None
        out_tensor = False
        img = np.asarray(image, dtype=np.float32)
        img_np = img

    if noise_typ.lower() in ("gauss", "gaussian"):
        gauss = np.random.normal(0.0, sigma, size=img_np.shape).astype(np.float32)
        noisy_np = img_np + gauss
    elif noise_typ.lower() == "poisson":
        # Ensure non-negative
        img_clip = np.clip(img_np, 0.0, None)
        # Poisson defined on counts; scale by peak then rescale back
        noisy_np = np.random.poisson(img_clip * float(peak)).astype(np.float32) / float(peak)
    else:
        raise ValueError(f"Unknown noise type: {noise_typ}")

    if is_normalize:
        noisy_np = normalize(noisy_np, data_type=np.float32)

    if is_tensor or out_tensor:
        noisy_t = torch.from_numpy(noisy_np).to(device=device, dtype=torch.float32)
        return noisy_t
    return noisy_np.astype(np.float32)


# -----------------------------
# Saving / I/O helpers
# -----------------------------

def save_image(image_numpy: np.ndarray, image_path: str, aspect_ratio: float = 1.0, save_all: bool = False):
    """
    Save a 2D image as PNG/TIFF or a 3D stack as multi-page TIFF (when save_all=True).

    Notes:
      - For 3D arrays and save_all=True, uses PIL multi-page TIFF (lossless).
      - For 3D arrays and save_all=False, saves a single mid-slice.
    """
    arr = np.asarray(image_numpy)
    if arr.ndim == 2:
        Image.fromarray(arr).save(image_path)
        return

    if arr.ndim == 3:
        if save_all:
            # PIL expects list of 2D frames (uint8/uint16 recommended)
            frames = [Image.fromarray(arr[i]) for i in range(arr.shape[0])]
            frames[0].save(image_path, save_all=True, append_images=frames[1:])
        else:
            mid = arr.shape[0] // 2
            Image.fromarray(arr[mid]).save(image_path)
        return

    # Fallback for >3D (e.g., CZYX): save first channel, mid-z
    if arr.ndim >= 4:
        zdim = -3  # assume (..., Z, Y, X)
        mid = arr.shape[zdim] // 2
        slice2d = np.take(arr, indices=mid, axis=zdim)
        # If still has channel dim, take first
        if slice2d.ndim > 2:
            slice2d = slice2d[0] if slice2d.shape[0] in (1, 3, 4) else np.squeeze(slice2d)
            if slice2d.ndim > 2:
                slice2d = slice2d[..., 0]
        Image.fromarray(slice2d).save(image_path)
        return


# -----------------------------
# Logging / Diagnostics
# -----------------------------

def print_numpy(x: np.ndarray, val: bool = True, shp: bool = False):
    """Print basic stats of a NumPy array."""
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        xf = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f'
              % (np.mean(xf), np.min(xf), np.max(xf), np.median(xf), np.std(xf)))


def diagnose_network(net: torch.nn.Module, name: str = 'network'):
    """
    Print mean absolute gradient across parameters (for quick exploding/vanishing checks).
    """
    total = 0.0
    count = 0
    for p in net.parameters():
        if p.grad is not None:
            total += float(torch.mean(torch.abs(p.grad.data)))
            count += 1
    mean = total / max(count, 1)
    print(name)
    print(mean)


# -----------------------------
# FS helpers
# -----------------------------

def mkdir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def mkdirs(paths):
    if isinstance(paths, (list, tuple)):
        for p in paths:
            mkdir(p)
    else:
        mkdir(paths)


# -----------------------------
# Dicing utilities (3D tiling)
# -----------------------------

def _steps_1d(length: int, roi: int, overlap: int) -> Tuple[int, int]:
    """
    Compute number of steps and trailing pad needed for 1D tiling.
    Returns: (n_steps, pad_after)
    """
    assert roi > overlap >= 0, "Require roi > overlap >= 0"
    step = roi - overlap
    # number of tiles so that coverage includes the tail
    n_steps = (length + overlap + step - 1) // step
    need = (n_steps - 1) * step + roi
    pad_after = max(0, need - length)
    return n_steps, pad_after


def pad_for_dicing(image: np.ndarray, roi_size: int, overlap: int = 0) -> np.ndarray:
    """
    Pad a 3D volume so that sliding cubes of size roi_size with given overlap
    tile the volume exactly.

    Returns padded image. Uses reflect padding.
    """
    assert image.ndim == 3, "pad_for_dicing expects (Z,Y,X)"
    Z, Y, X = image.shape
    nz, pad_z = _steps_1d(Z, roi_size, overlap)
    ny, pad_y = _steps_1d(Y, roi_size, overlap)
    nx, pad_x = _steps_1d(X, roi_size, overlap)

    npad = ((0, pad_z), (0, pad_y), (0, pad_x))
    image_padded = np.pad(image, pad_width=npad, mode='reflect')
    print(f"image volume padded for equal dicing. pad sizes: {npad} -> tiles (Z,Y,X)=({nz},{ny},{nx})")
    return image_padded


def crop_for_dicing(image: np.ndarray, roi_size: int, overlap: int = 0) -> np.ndarray:
    """
    Crop a 3D volume (from the beginning) so that sliding cubes tile exactly.
    Useful when you prefer cropping over padding.
    """
    assert image.ndim == 3, "crop_for_dicing expects (Z,Y,X)"
    Z, Y, X = image.shape
    step = roi_size - overlap
    # how many steps fully fit
    nz = max(1, (Z - overlap) // step)
    ny = max(1, (Y - overlap) // step)
    nx = max(1, (X - overlap) // step)
    needZ = (nz - 1) * step + roi_size
    needY = (ny - 1) * step + roi_size
    needX = (nx - 1) * step + roi_size
    z_crop = max(0, Z - needZ)
    y_crop = max(0, Y - needY)
    x_crop = max(0, X - needX)
    img_cropped = image[z_crop:, y_crop:, x_crop:]
    print(f"image volume cropped for equal dicing. crop sizes: {(z_crop, y_crop, x_crop)} -> tiles (Z,Y,X)=({nz},{ny},{nx})")
    return img_cropped
