# data/__init__.py  — robust loader for 3D patch training (CycleGAN project)
# Updated: keeps your public API (find_dataset_using_name, get_option_setter, create_dataset)

import importlib
import types
import numpy as np
import torch
import torch.utils.data as tud
from data.base_dataset import BaseDataset


def _import_dataset_module(dataset_name: str):
    """Import module 'data/[dataset_name]_dataset.py' with helpful error messages."""
    module_name = f"data.{dataset_name}_dataset"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"Could not import dataset module '{module_name}'. "
            f"Make sure 'data/{dataset_name}_dataset.py' exists."
        ) from e


def find_dataset_using_name(dataset_name: str):
    """
    Import the module 'data/[dataset_name]_dataset.py' and return the dataset class
    '[DatasetName]Dataset' (case-insensitive, underscores ignored) that subclasses BaseDataset.
    """
    datasetlib = _import_dataset_module(dataset_name)

    dataset = None
    target = dataset_name.replace('_', '') + 'dataset'  # e.g., 'ot_lsm' -> 'otlsmdataset'
    for name, cls in datasetlib.__dict__.items():
        if isinstance(cls, type) and issubclass(cls, BaseDataset):
            if name.lower() == target.lower():
                dataset = cls
                break

    if dataset is None:
        raise NotImplementedError(
            f"In data/{dataset_name}_dataset.py, define a subclass of BaseDataset named "
            f"'{dataset_name.replace('_','').title()}Dataset' (case-insensitive)."
        )
    return dataset


def get_option_setter(dataset_name: str):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """
    Main entry: wraps CustomDatasetDataLoader.
      >>> from data import create_dataset
      >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    return data_loader.load_data()


def _seed_worker(worker_id):
    """
    Make dataloading deterministic-ish across workers.
    Respects torch initial seed; propagates to numpy/python RNG.
    """
    worker_seed = torch.initial_seed() % 2**31
    np.random.seed(worker_seed)


class CustomDatasetDataLoader:
    """Wrapper class of Dataset that performs multi-threaded data loading with sane defaults."""

    def __init__(self, opt):
        """
        Step 1: create dataset instance given '--dataset_mode'
        Step 2: configure Sampler (Distributed or not) and DataLoader
        """
        self.opt = opt

        # ---- Create dataset instance
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print(f"dataset [{type(self.dataset).__name__}] was created")

        # ---- Options with safe fallbacks
        batch_size = getattr(opt, "batch_size", 1)
        num_workers = int(getattr(opt, "num_threads", 0))
        serial_batches = bool(getattr(opt, "serial_batches", False))
        drop_last = bool(getattr(opt, "drop_last", getattr(opt, "isTrain", True)))
        pin_memory = bool(getattr(opt, "pin_memory", True))
        prefetch_factor = getattr(opt, "prefetch_factor", 2) if num_workers > 0 else None

        # Distributed?
        world_size = int(getattr(opt, "world_size", 1))
        rank = int(getattr(opt, "rank", 0))
        use_distributed = bool(getattr(opt, "distributed", False)) or world_size > 1

        # ---- Sampler
        sampler = None
        if use_distributed:
            sampler = tud.distributed.DistributedSampler(
                self.dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=not serial_batches,
                drop_last=drop_last,
            )
            shuffle = False  # sampler controls shuffling
        else:
            sampler = None
            shuffle = not serial_batches

        # Allow dataset to provide a custom collate_fn if needed (e.g., variable-size 3D crops)
        collate_fn = getattr(self.dataset, "collate_fn", None)
        if not isinstance(collate_fn, (types.FunctionType, types.MethodType)):
            collate_fn = None

        # ---- DataLoader
        self.dataloader = tud.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            worker_init_fn=_seed_worker,
            persistent_workers=(num_workers > 0),
            prefetch_factor=prefetch_factor,
            collate_fn=collate_fn,
        )

        self._sampler = sampler

    # API kept identical to your original:

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of items *available* through this loader (respecting max_dataset_size)."""
        raw = getattr(self.opt, "max_dataset_size", None)
        # Treat None or +/-inf as "no explicit cap"
        if raw is None or (isinstance(raw, float) and np.isinf(raw)):
            max_sz = len(self.dataset)
        else:
            max_sz = int(raw)
        return min(len(self.dataset), max_sz)

    def __iter__(self):
        """
        Yield batches while respecting max_dataset_size.
        If distributed, set sampler epoch if 'epoch' exists in opt.
        """
        if self._sampler is not None and hasattr(self._sampler, "set_epoch"):
            # makes shuffling different each epoch in DDP
            epoch = int(getattr(self.opt, "epoch", 0))
            self._sampler.set_epoch(epoch)

        bsz = int(getattr(self.opt, "batch_size", 1))
        raw = getattr(self.opt, "max_dataset_size", None)
        if raw is None or (isinstance(raw, float) and np.isinf(raw)):
            max_sz = len(self.dataset)
        else:
            max_sz = int(raw)
        max_batches = (max_sz + bsz - 1) // bsz  # ceil

        for i, data in enumerate(self.dataloader):
            if i >= max_batches:
                break
            yield data
