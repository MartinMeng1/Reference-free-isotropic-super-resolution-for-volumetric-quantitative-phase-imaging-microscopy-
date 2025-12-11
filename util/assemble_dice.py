# util/assemble_dice.py  — streaming version (no huge cube_queue)

import numpy as np
from collections import OrderedDict
from skimage.exposure import match_histograms
from skimage import exposure
import data


class Assemble_Dice:
    """
    Re-assembles a 3D volume from overlapped sub-volumes ("dice") in a streaming way.

    Key points:
      - No cube_queue: cubes are merged into the big volume as they arrive.
      - mask_ret is uint8 to save memory.
      - Currently assembles 'fake' and 'rec' (no 'real' to save RAM).
    """

    def __init__(self, opt):
        # Probe dataset for original/padded sizes
        dataset_class = data.find_dataset_using_name(opt.dataset_mode)
        probe = dataset_class(opt)
        self.image_size_original = probe.size_original()
        self.image_size = probe.size()
        self.border_cut = int(getattr(opt, "border_cut", 0))

        # Dicing geometry
        self.roi_size = int(opt.dice_size[0])
        self.overlap = int(getattr(opt, "overlap", 0))
        self.step = self.roi_size - self.overlap
        assert self.step > 0, "dice_size must be > overlap."

        self.z_steps = (self.image_size[0] - self.overlap) // self.step
        self.y_steps = (self.image_size[1] - self.overlap) // self.step
        self.x_steps = (self.image_size[2] - self.overlap) // self.step
        self.num_cubes = self.z_steps * self.y_steps * self.x_steps

        # What to assemble: fake (G_A output) and rec (G_B(G_A) reconstruction)
        self.visual_names = ["fake", "rec"]
        self.skip_real = bool(getattr(opt, "skip_real", False))

        # Output storages (float32 accumulators) + uint8 mask
        self.visual_ret = OrderedDict()
        self.mask_ret = OrderedDict()
        self.current_idx = {}  # streaming cube index per name

        for name in self.visual_names:
            self.visual_ret[name] = np.zeros(self.image_size, dtype=np.float32)
            self.mask_ret[name] = np.zeros(self.image_size, dtype=np.uint8)
            self.current_idx[name] = 0

        # Post-processing switches
        self.histogram_match = bool(getattr(opt, "histogram_match", False))
        self.normalize_intensity = bool(getattr(opt, "normalize_intensity", False))
        if self.normalize_intensity:
            self.p1, self.p99 = getattr(opt, "sat_level", (1.0, 99.0))

        # Output dtype selection: 'uint8' | 'uint16' | 'float'
        self.imtype = getattr(opt, "data_type", "uint16")

    # ---------- indexing helpers ----------
    def indexTo3DIndex(self, index):
        x_idx = index % self.x_steps
        y_idx = (index % (self.x_steps * self.y_steps)) // self.x_steps
        z_idx = index // (self.x_steps * self.y_steps)
        return z_idx, y_idx, x_idx

    def indexToCoordinates(self, index):
        z_idx, y_idx, x_idx = self.indexTo3DIndex(index)
        cz = z_idx * self.step
        cy = y_idx * self.step
        cx = x_idx * self.step
        return cz, cy, cx

    # ---------- main API ----------
    def addToStack(self, cube):
        """
        cube: OrderedDict with keys in self.visual_ret (e.g., 'fake','rec'),
              each a torch tensor of shape (B=1, C=1, Z, Y, X) in [0,1].
        We STREAM: directly merge each cube into the output volume.
        """
        for name in self.visual_ret.keys():
            if name not in cube:
                # model might not provide 'rec' for some configs
                continue

            image_tensor = cube[name]
            np_cube = image_tensor.detach().cpu().float().numpy().squeeze()  # (Z,Y,X)
            if np_cube.ndim != 3:
                raise ValueError(f"{name} cube must be 3D (Z,Y,X), got shape {np_cube.shape}")

            # Optional border cut
            if self.border_cut > 0:
                bc = self.border_cut
                np_cube = np_cube[bc:-bc, bc:-bc, bc:-bc]

            if np_cube.shape != (self.roi_size, self.roi_size, self.roi_size):
                raise ValueError(f"{name} cube has shape {np_cube.shape}, expected {(self.roi_size,)*3}")

            out = self.visual_ret[name]
            mask = self.mask_ret[name]
            idx = self.current_idx[name]

            if idx >= self.num_cubes:
                raise RuntimeError(f"Received more cubes than expected: idx={idx}, num_cubes={self.num_cubes}")

            cz, cy, cx = self.indexToCoordinates(idx)
            zslice = slice(cz, cz + self.roi_size)
            yslice = slice(cy, cy + self.roi_size)
            xslice = slice(cx, cx + self.roi_size)

            out[zslice, yslice, xslice] += np_cube.astype(np.float32)
            mask[zslice, yslice, xslice] += 1  # uint8 is enough for overlap count here

            self.current_idx[name] += 1

    def assemble_all(self):
        """
        Finalize assembly:
          - divide by mask
          - optional intensity normalization
          - optional histogram match (fake->real, if real ever added)
          - crop padding
          - cast dtype
        """
        for name in self.visual_ret.keys():
            print(f"Patching for... {name}")
            out = self.visual_ret[name]
            mask = self.mask_ret[name]

            nonzero = mask > 0
            if not np.any(nonzero):
                raise RuntimeError(f"No voxels written for {name}.")

            out[nonzero] = out[nonzero] / mask[nonzero]
            print(f"done patching the cubes for {name} image volume.")
            print(f"Max overlap count for {name}: {mask.max()}")

            # Optional normalization
            if self.normalize_intensity:
                p1_, p99_ = np.percentile(out[nonzero], (self.p1, self.p99))
                out[:] = exposure.rescale_intensity(out, in_range=(p1_, p99_))

            # Optional histogram matching (if we ever include 'real')
            if self.histogram_match and "real" in self.visual_ret and name == "fake":
                try:
                    out[:] = match_histograms(out, self.visual_ret["real"])
                except Exception as e:
                    print(f"[warn] histogram_match failed: {e}")

            # Crop padding back to original size
            if self.image_size_original is not None:
                pads = [self.image_size[i] - self.image_size_original[i] for i in range(3)]
                print(f"Image cropped back to original size by: {pads}")
                zc, yc, xc = pads
                if zc > 0:
                    out = out[:-zc]
                if yc > 0:
                    out = out[:, :-yc]
                if xc > 0:
                    out = out[:, :, :-xc]
                self.visual_ret[name] = out  # replace with cropped

            # Final dtype cast
            self.visual_ret[name] = self._cast_imtype(self.visual_ret[name], self.imtype)

    # ---------- accessors ----------
    def getSnapshots(self, index, slice_axis=2):
        snaps = OrderedDict()
        for name, vol in self.visual_ret.items():
            if slice_axis == 0:
                snaps[name] = vol[index, :, :]
            elif slice_axis == 1:
                snaps[name] = vol[:, index, :]
            else:
                snaps[name] = vol[:, :, index]
        return snaps

    def getDict(self):
        return self.visual_ret

    def getMaskRet(self):
        key = next(iter(self.mask_ret))
        return self.mask_ret[key]

    # ---------- utils ----------
    @staticmethod
    def _cast_imtype(arr: np.ndarray, imtype: str) -> np.ndarray:
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)
        if imtype == "uint8":
            return (arr * 255.0 + 0.5).astype(np.uint8)
        elif imtype == "uint16":
            return (arr * (2**16 - 1) + 0.5).astype(np.uint16)
        elif imtype == "float":
            return arr.astype(np.float32)
        else:
            raise ValueError(f"Unknown data_type '{imtype}'. Use 'uint8'|'uint16'|'float'.")
