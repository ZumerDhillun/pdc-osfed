"""CIFAR-10 loading (Step 4).

Design (fast and light on RAM):
  * The dataset is downloaded once with torchvision, then cached as uint8 tensors
    in data/raw/cifar10_uint8.pt (about 180 MB in total).
  * Images stay uint8 in memory. They are normalized batch by batch with the
    standard CIFAR-10 mean and std (decision D10). No augmentation.
  * FastLoader replaces torch DataLoader: no PIL, no worker processes, so it is
    much faster on a 4-core CPU and easy to copy into client processes later.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from src.utils import project_root

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10
IMAGE_SHAPE = (3, 32, 32)

_MEAN = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)


@dataclass
class CifarData:
    x_train: torch.Tensor  # uint8, shape (50000, 3, 32, 32)
    y_train: torch.Tensor  # int64, shape (50000,)
    x_test: torch.Tensor   # uint8, shape (10000, 3, 32, 32)
    y_test: torch.Tensor   # int64, shape (10000,)


def load_cifar10(data_dir: Optional[str | Path] = None, download: bool = True) -> CifarData:
    """Load CIFAR-10 as uint8 tensors. The first call downloads it and writes a cache."""
    data_dir = Path(data_dir) if data_dir else project_root() / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "cifar10_uint8.pt"
    if cache.exists():
        blob = torch.load(cache, map_location="cpu", weights_only=True)
        return CifarData(**blob)

    from torchvision.datasets import CIFAR10  # imported here so tests work without torchvision

    train = CIFAR10(root=str(data_dir), train=True, download=download)
    test = CIFAR10(root=str(data_dir), train=False, download=download)

    def to_tensors(ds):
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()  # HWC -> CHW, uint8
        y = torch.tensor(ds.targets, dtype=torch.long)
        return x, y

    x_train, y_train = to_tensors(train)
    x_test, y_test = to_tensors(test)
    blob = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
    torch.save(blob, cache)
    return CifarData(**blob)


def normalize_batch(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 (B,3,32,32) -> float32 normalized with the CIFAR-10 mean and std."""
    x = x_uint8.float().div_(255.0)
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


class FastLoader:
    """Minimal in-memory batch iterator.

    x: uint8 tensor (N,3,32,32), y: int64 tensor (N,)
    indices: optional subset (for example one client's partition)
    device: if given, the subset is moved there once (useful on a GPU)
    The shuffle order is reproducible: epoch e uses seed + e.
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        indices: Optional[Sequence[int]] = None,
        batch_size: int = 64,
        shuffle: bool = True,
        seed: int = 0,
        device: Optional[torch.device] = None,
    ):
        if indices is not None:
            idx = torch.as_tensor(np.asarray(indices), dtype=torch.long)
            x, y = x[idx], y[idx]
        if device is not None:
            x, y = x.to(device), y.to(device)
        self.x, self.y = x, y
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)
        self._epoch = 0

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])

    def __len__(self) -> int:
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        n = self.n_samples
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch)
            order = torch.randperm(n, generator=g).to(self.x.device)
        else:
            order = torch.arange(n, device=self.x.device)
        self._epoch += 1
        for start in range(0, n, self.batch_size):
            sel = order[start:start + self.batch_size]
            yield normalize_batch(self.x[sel]), self.y[sel]
