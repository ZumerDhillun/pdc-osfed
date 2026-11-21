"""Small shared helpers: paths, seeding, device, environment report."""
from __future__ import annotations

import os
import platform
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def project_root() -> Path:
    """Folder that contains src/, configs/, data/, ... (works on Ubuntu and Kaggle)."""
    return Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def today_stamp() -> str:
    """YYYYMMDD, used as the prefix of run IDs."""
    return datetime.now().strftime("%Y%m%d")


def env_report() -> str:
    """Text describing the machine and library versions (save it in notes/env_*.txt)."""
    lines = [
        f"date: {datetime.now().isoformat(timespec='seconds')}",
        f"python: {sys.version.split()[0]}",
        f"platform: {platform.platform()}",
        f"cpu_count: {os.cpu_count()}",
        f"numpy: {np.__version__}",
        f"torch: {torch.__version__}",
    ]
    try:
        import torchvision

        lines.append(f"torchvision: {torchvision.__version__}")
    except Exception:
        lines.append("torchvision: not installed")
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        lines.append(f"ram_gb: {ram_gb:.1f}")
    except (ValueError, OSError, AttributeError):
        lines.append("ram_gb: unknown")
    lines.append(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"gpu: {torch.cuda.get_device_name(0)}")
    return "\n".join(lines) + "\n"
