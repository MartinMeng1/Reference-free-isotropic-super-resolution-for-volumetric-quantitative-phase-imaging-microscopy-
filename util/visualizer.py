# TODO Sep 08 version (enhanced MIP visualization)
import os
import ntpath
import time
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from tifffile import imwrite  # robust for 3D stacks
import torch

from . import util, html


# ---------------------------
# Utilities
# ---------------------------
def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _central_slices(vol_np: np.ndarray):
    """Return (xy, xz, yz) central slices from a 3D ndarray (Z,Y,X)."""
    assert vol_np.ndim == 3, f"central_slices expects 3D, got {vol_np.shape}"
    z, y, x = vol_np.shape
    return (
        vol_np[z // 2, :, :],   # XY @ mid Z
        vol_np[:, y // 2, :],   # XZ @ mid Y
        vol_np[:, :, x // 2],   # YZ @ mid X
    )


def _mips(vol_np: np.ndarray):
    """Return (mip_xy, mip_xz, mip_yz) max intensity projections from 3D ndarray (Z,Y,X)."""
    assert vol_np.ndim == 3, f"mips expects 3D, got {vol_np.shape}"
    mip_xy = np.max(vol_np, axis=0)  # collapse Z -> XY
    mip_xz = np.max(vol_np, axis=1)  # collapse Y -> XZ
    mip_yz = np.max(vol_np, axis=2)  # collapse X -> YZ
    return mip_xy, mip_xz, mip_yz


def _clip_to_uint8(arr: np.ndarray, p1=1, p99=99) -> np.ndarray:
    """
    Percentile clip -> [0,255] uint8 for display.
    Keeps visualization robust to outliers (prevents 'washed-out' MIPs).
    """
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, [p1, p99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # fallback: min-max
        mn, mx = np.min(a), np.max(a)
        if mx <= mn:
            return np.zeros_like(a, dtype=np.uint8)
        a = (a - mn) / (mx - mn + 1e-8)
        return (a * 255).astype(np.uint8)

    a = np.clip(a, lo, hi)
    a = (a - lo) / (hi - lo + 1e-8)
    return (a * 255).astype(np.uint8)


def _gamma_u8(arr_u8: np.ndarray, gamma=0.9) -> np.ndarray:
    """
    Optional gamma after clipping for additional contrast (gamma<1 brightens midtones).
    Input/Output: uint8.
    """
    a = arr_u8.astype(np.float32) / 255.0
    a = np.power(a, gamma)
    return (a * 255).astype(np.uint8)


# ---------------------------
# HTML save (legacy helper)
# ---------------------------
def save_images(webpage, visuals: OrderedDict, image_path, aspect_ratio=1.0, width=256):
    """
    Save images to disk and add to HTML page.
    Handles 2D or 3D volumes.
    - For 3D: writes a full .tif stack and also a PNG of the central XY slice for quick preview.
    - For 2D: writes a PNG.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        # Convert tensor -> numpy; keep high bit-depth for TIFF
        image_np = util.tensor2im(im_data, imtype=np.uint16).squeeze()

        label_dir = os.path.join(image_dir, label)
        _ensure_dir(label_dir)

        if image_np.ndim == 3:
            # Save full volume stack
            tiff_name = f'{name}_{label}.tif'
            tiff_path = os.path.join(label_dir, tiff_name)
            imwrite(tiff_path, image_np, photometric='minisblack')

            # Also save a central XY slice preview (uint8)
            xy, _, _ = _central_slices(image_np)
            png_name = f'{name}_{label}_xy.png'
            png_path = os.path.join(label_dir, png_name)
            # use percentile clipping for preview consistency
            xy_u8 = _clip_to_uint8(xy, 1, 99)
            util.save_image(xy_u8, png_path, aspect_ratio=aspect_ratio)

            ims.append(png_path)
            txts.append(label + " (XY)")
            links.append(tiff_path)  # click through to full stack
        elif image_np.ndim == 2:
            png_name = f'{name}_{label}.png'
            png_path = os.path.join(label_dir, png_name)
            png_u8 = _clip_to_uint8(image_np, 1, 99)
            util.save_image(png_u8, png_path, aspect_ratio=aspect_ratio)
            ims.append(png_path)
            txts.append(label)
            links.append(png_path)
        else:
            raise ValueError(f"Expected 2D/3D array after squeeze, got shape {image_np.shape} for {label}")

    webpage.add_images(ims, txts, links, width=width)


def save_test_metrics(save_dir, opt, ssims, psnrs):
    ssim_avg_input_gt, ssim_avg_output_gt, ssim_whole_input_gt, ssim_whole_output_gt = ssims
    psnr_avg_input_gt, psnr_avg_output_gt, psnr_whole_input_gt, psnr_whole_output_gt = psnrs

    message = (
        f'Experiment Name: {opt.name}\n'
        '-------------------------------------------------\n'
        'Network Input vs. Groundtruth\n'
        f'(ssim_avg: {ssim_avg_input_gt:.4f}, psnr_avg: {psnr_avg_input_gt:.4f}, '
        f'ssim_whole: {ssim_whole_input_gt:.4f}, psnr_whole: {psnr_whole_input_gt:.4f})\n'
        '-------------------------------------------------\n'
        'Network Output vs. Groundtruth\n'
        f'(ssim_avg: {ssim_avg_output_gt:.4f}, psnr_avg: {psnr_avg_output_gt:.4f}, '
        f'ssim_whole: {ssim_whole_output_gt:.4f}, psnr_whole: {psnr_whole_output_gt:.4f})\n'
        '-------------------------------------------------'
    )

    print(message)
    filename = os.path.join(save_dir, 'metrics.txt')
    with open(filename, "a") as f:
        f.write(message + '\n')


# ---------------------------
# Visualizer (TensorBoard + disk)
# ---------------------------
class Visualizer:
    def __init__(self, opt):
        self.opt = opt
        self.win_size = opt.display_winsize
        self.use_html = opt.isTrain and not opt.no_html
        self.name = opt.name
        self.port = opt.display_port
        self.display_histogram = opt.display_histogram

        self.saved = False

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print(f'create web directory {self.web_dir}...')
            util.mkdirs([self.web_dir, self.img_dir])

        self.tb_dir = os.path.join(opt.checkpoints_dir, 'tensorboard')
        print(f'create tensorboard directory {self.tb_dir}...')
        util.mkdir(self.tb_dir)

        from torch.utils.tensorboard import SummaryWriter
        self.log_dir = os.path.join(self.tb_dir, self.name)
        self.tb_writer = SummaryWriter(self.log_dir)

        # training loss log file
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(f'================ Training Loss ({now}) ================\n')

        # visualization tuning
        self.mip_p1 = 1
        self.mip_p99 = 99
        self.mip_gamma = 0.9  # set to 1.0 to disable gamma

    def reset(self):
        self.saved = False

    # ---------- figures ----------
    def _figure_from_slices(self, vol_uint8: np.ndarray, title_prefix=""):
        """Return a figure with central slices (XY/XZ/YZ)."""
        xy, xz, yz = _central_slices(vol_uint8)
        # ensure proper range
        xy = _clip_to_uint8(xy, 1, 99)
        xz = _clip_to_uint8(xz, 1, 99)
        yz = _clip_to_uint8(yz, 1, 99)

        fig = plt.figure(dpi=150)
        ax1 = fig.add_subplot(1, 3, 1); ax1.set_axis_off(); ax1.set_title(f'{title_prefix}XY')
        ax2 = fig.add_subplot(1, 3, 2); ax2.set_axis_off(); ax2.set_title(f'{title_prefix}XZ')
        ax3 = fig.add_subplot(1, 3, 3); ax3.set_axis_off(); ax3.set_title(f'{title_prefix}YZ')
        ax1.imshow(xy, cmap='gray', vmin=0, vmax=255)
        ax2.imshow(xz, cmap='gray', vmin=0, vmax=255)
        ax3.imshow(yz, cmap='gray', vmin=0, vmax=255)
        fig.tight_layout(pad=0.2)
        return fig

    def _figure_from_mips(self, vol_uint8: np.ndarray, title_prefix=""):
        """Return a figure with MIPs (XY/XZ/YZ) using percentile clipping (and optional gamma)."""
        mip_xy, mip_xz, mip_yz = _mips(vol_uint8)

        # robust display: percentile clip + optional gamma
        mip_xy = _clip_to_uint8(mip_xy, self.mip_p1, self.mip_p99)
        mip_xz = _clip_to_uint8(mip_xz, self.mip_p1, self.mip_p99)
        mip_yz = _clip_to_uint8(mip_yz, self.mip_p1, self.mip_p99)

        if self.mip_gamma and self.mip_gamma != 1.0:
            mip_xy = _gamma_u8(mip_xy, self.mip_gamma)
            mip_xz = _gamma_u8(mip_xz, self.mip_gamma)
            mip_yz = _gamma_u8(mip_yz, self.mip_gamma)

        fig = plt.figure(dpi=150)
        ax1 = fig.add_subplot(1, 3, 1); ax1.set_axis_off(); ax1.set_title(f'{title_prefix}XY MIP')
        ax2 = fig.add_subplot(1, 3, 2); ax2.set_axis_off(); ax2.set_title(f'{title_prefix}XZ MIP')
        ax3 = fig.add_subplot(1, 3, 3); ax3.set_axis_off(); ax3.set_title(f'{title_prefix}YZ MIP')
        ax1.imshow(mip_xy, cmap='gray', vmin=0, vmax=255)
        ax2.imshow(mip_xz, cmap='gray', vmin=0, vmax=255)
        ax3.imshow(mip_yz, cmap='gray', vmin=0, vmax=255)
        fig.tight_layout(pad=0.2)
        return fig

    # ---------- main TB display ----------
    def display_current_results(self, visuals: OrderedDict, epoch: int):
        """
        Show results in TensorBoard.
        - For (B,C,Z,Y,X) tensors in [0,1], log central slices & MIPs of the first item.
        - For classifier mode, render labels.
        """
        for label, image in visuals.items():
            if self.opt.model != 'classifier':
                # Expect (B,C,Z,Y,X); convert to uint8 volume for display
                img_np = util.tensor2im(image, imtype=np.uint8)  # scales to 0..255 internally
                vol = img_np[0, 0]  # (Z,Y,X)
                if vol.ndim != 3:
                    raise ValueError(f"Expected 3D volume for TB display, got {vol.shape}")

                fig_slice = self._figure_from_slices(vol, title_prefix=f'{label} ')
                self.tb_writer.add_figure('train_slice_images/' + label, fig_slice, epoch)
                plt.close(fig_slice)

                fig_mip = self._figure_from_mips(vol, title_prefix=f'{label} ')
                self.tb_writer.add_figure('train_mip_images/' + label, fig_mip, epoch)
                plt.close(fig_mip)

            else:
                # Classifier display path
                if label in ('output_tr_softmax', 'output_val_softmax', 'label_GT'):
                    predicted = torch.argmax(image[0]).item()
                    label_print_str = 'Axial' if predicted == 0 else 'Lateral'
                    fig = plt.figure(dpi=120)
                    plt.text(0.1, 0.4, label_print_str, fontsize=36,
                             bbox=dict(boxstyle="square", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
                    self.tb_writer.add_figure('train_images/' + label, fig, epoch)
                    plt.close(fig)
                else:
                    img_np = util.tensor2im(image[0], imtype=np.uint8).squeeze()
                    img_u8 = _clip_to_uint8(img_np, 1, 99)
                    fig = plt.figure(dpi=120)
                    plt.imshow(img_u8, cmap='gray', vmin=0, vmax=255); plt.axis('off')
                    self.tb_writer.add_figure('train_images/' + label, fig, epoch)
                    plt.close(fig)

    def display_model_hyperparameters(self):
        # shows as markdown in TensorBoard
        message = '--------------- Options ------------------  \n'
        for k, v in sorted(vars(self.opt).items()):
            message += f'**{k}**: {v}  \n'
        message += '----------------- End -------------------'
        self.tb_writer.add_text('Model_hyperparameters', message)

    def display_current_histogram(self, visuals, epoch):
        if not self.display_histogram:
            return
        for label, image in visuals.items():
            try:
                self.tb_writer.add_histogram('train_histograms/' + label, image[0][0], epoch)
            except Exception as e:
                print(f"[warn] histogram for {label} skipped: {e}")

    def display_graph(self, model, visuals):
        try:
            example = next(iter(visuals.values()))
            self.tb_writer.add_graph(model, example)
        except Exception as e:
            print(f"[warn] add_graph skipped: {e}")

    # ---------- disk save ----------
    def save_current_visuals(self, visuals: OrderedDict, epoch: int):
        """
        Saves per-label outputs under <checkpoints>/<name>/web/images as TIFF stacks.
        Also writes PNG of central XY slice with clipping for preview.
        """
        if not hasattr(self, 'img_dir'):
            return
        _ensure_dir(self.img_dir)

        for label, image in visuals.items():
            vol_u16 = util.tensor2im(image[0], imtype=np.uint16).squeeze()

            if vol_u16.ndim == 3:
                # Save TIFF stack
                tiff_path = os.path.join(self.img_dir, f'{epoch}_{label}.tif')
                imwrite(tiff_path, vol_u16, photometric='minisblack')

                # Save central XY preview PNG with clipping
                xy, _, _ = _central_slices(vol_u16)
                xy_u8 = _clip_to_uint8(xy, self.mip_p1, self.mip_p99)
                if self.mip_gamma and self.mip_gamma != 1.0:
                    xy_u8 = _gamma_u8(xy_u8, self.mip_gamma)
                png_path = os.path.join(self.img_dir, f'{epoch}_{label}_xy.png')
                util.save_image(xy_u8, png_path)

            elif vol_u16.ndim == 2:
                # 2D: just PNG
                png_u8 = _clip_to_uint8(vol_u16, 1, 99)
                png_path = os.path.join(self.img_dir, f'{epoch}_{label}.png')
                util.save_image(png_u8, png_path)
            else:
                print(f"[warn] save_current_visuals: unexpected ndim {vol_u16.ndim} for {label}")

    # ---------- scalars / logs ----------
    def plot_current_losses(self, plot_count, losses: OrderedDict, is_epoch=False):
        for label, loss in losses.items():
            tag = 'train_by_epoch/' + label if is_epoch else 'train_by_epoch_progress/' + label
            self.tb_writer.add_scalar(tag, loss, plot_count)

    def print_current_losses(self, epoch, epoch_progress, losses: OrderedDict, t_comp, t_data):
        message = f'(epoch: {epoch}, epoch_progress: {epoch_progress}%, iter time: {t_comp:.3f}, data load time: {t_data:.3f}) '
        for k, v in losses.items():
            message += f'{k}: {v:.3f} '
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write(message + '\n')
