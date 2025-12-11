# TODO August 06 version (CUDA-always refactor)
import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class BaseModel(ABC):
    """
    Abstract base class for models.
    Subclasses must implement: __init__, set_input, forward, optimize_parameters.
    """

    def __init__(self, opt):
        """
        Parameters
        ----------
        opt : options object
            Experiment options
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.dimension = getattr(opt, "image_dimension", 2)

        # CUDA is assumed to be available; use the first GPU id
        assert torch.cuda.is_available(), "CUDA is required but not available."
        assert len(self.gpu_ids) > 0 and self.gpu_ids[0] >= 0, \
            "Provide at least one valid GPU id."
        self.device = torch.device(f'cuda:{self.gpu_ids[0]}')

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

        # Fastest algo selection for fixed-size inputs (typical for 3D training)
        torch.backends.cudnn.benchmark = True

        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for 'plateau' scheduler

        # NEW: keep track of total iterations (restored from checkpoints)
        self.total_iters = 0

    # ---- hooks to customize options ----
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    # ---- abstract API ----
    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    # ---- lifecycle ----
    def setup(self, opt):
        """Create schedulers; optionally load checkpoints; print nets."""
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optim, opt)
                               for optim in self.optimizers]
        if (not self.isTrain) or opt.continue_train:
            load_suffix = f'iter_{opt.load_iter}' if opt.load_iter > 0 else str(opt.epoch)
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                getattr(self, 'net' + name).eval()

    @torch.no_grad()
    def test(self):
        self.forward()
        self.compute_visuals()

    def compute_visuals(self):
        pass

    # ---- reporting ----
    def get_image_paths(self):
        return self.image_paths

    def update_learning_rate(self):
        """Call once per epoch."""
        for scheduler in getattr(self, "schedulers", []):
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

    def get_current_visuals(self):
        out = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                out[name] = getattr(self, name)
        return out

    def get_current_losses(self):
        out = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                out[name] = float(getattr(self, 'loss_' + name))
        return out

    # ---- iteration tracking helper ----
    def set_total_iters(self, n: int):
        """Called from the training loop to keep iteration counter in sync."""
        self.total_iters = int(n)

    # ---- checkpointing ----
    def save_networks(self, epoch):
        """Save all nets to disk. Works with or without DataParallel."""
        for name in self.model_names:
            if not isinstance(name, str):
                continue
            filename = f'{epoch}_net_{name}.pth'
            path = os.path.join(self.save_dir, filename)

            net = getattr(self, 'net' + name)
            net_to_save = net.module if isinstance(net, torch.nn.DataParallel) else net

            # Save CPU state dict (portable), then restore device
            state = net_to_save.state_dict()
            torch.save(state, path)

        # ALSO save a tiny training-state file with iteration counter
        try:
            meta = {'total_iters': int(getattr(self, 'total_iters', 0))}
            meta_path = os.path.join(self.save_dir, f'{epoch}_training_state.pth')
            torch.save(meta, meta_path)
        except Exception:
            # If anything goes wrong here, we don't want to kill training
            pass

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints (pre-0.4)."""
        key = keys[i]
        if i + 1 == len(keys):
            if module.__class__.__name__.startswith('InstanceNorm'):
                if key in ('running_mean', 'running_var') and getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
                if key == 'num_batches_tracked':
                    state_dict.pop('.'.join(keys), None)
        else:
            self.__patch_instance_norm_state_dict(
                state_dict, getattr(module, key), keys, i + 1
            )

    def load_networks(self, epoch):
        """Load all nets from disk (keeps DP wrapping if present)."""
        for name in self.model_names:
            if not isinstance(name, str):
                continue
            filename = f'{epoch}_net_{name}.pth'
            path = os.path.join(self.save_dir, filename)
            print(f'loading the model from {path}')

            net = getattr(self, 'net' + name)
            is_dp = isinstance(net, torch.nn.DataParallel)
            target = net.module if is_dp else net

            state_dict = torch.load(path, map_location=self.device)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            for k in list(state_dict.keys()):
                self.__patch_instance_norm_state_dict(state_dict, target, k.split('.'))

            target.load_state_dict(state_dict, strict=True)
            net.to(self.device)  # keep outer wrapper on device

        # try to load training-state metadata (total_iters, etc.)
        meta_path = os.path.join(self.save_dir, f'{epoch}_training_state.pth')
        if os.path.exists(meta_path):
            try:
                meta = torch.load(meta_path, map_location='cpu')
                self.total_iters = int(meta.get('total_iters', 0))
                print(f'[info] restored total_iters = {self.total_iters}')
            except Exception:
                print('[warn] could not load training_state meta; '
                      'starting total_iters from 0.')
                self.total_iters = 0

    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if not isinstance(name, str):
                continue
            net = getattr(self, 'net' + name)
            n_params = sum(p.numel() for p in net.parameters())
            if verbose:
                print(net)
            print(f'[Network {name}] Total number of parameters : '
                  f'{n_params/1e6:.3f} M')
        print('---------------------------------------------')

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Enable/disable grad for all nets in `nets`."""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is None:
                continue
            for p in net.parameters():
                p.requires_grad = requires_grad
