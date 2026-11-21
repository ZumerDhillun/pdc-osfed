"""Bytes, evaluation, timing and CSV logging.

Byte counting rule (decision D17):
  * payload_bytes(obj): bytes of the tensor data only (numel x element size). This is the MAIN number.
  * wire_bytes(obj):    size of the serialized message (torch.save). Logged as a second number.
  The same two functions are used for every message of both methods.
"""
from __future__ import annotations

import csv
import io
import time
from pathlib import Path
from typing import Iterable, Iterator

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- byte counting


def _iter_tensors(obj) -> Iterator[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_tensors(v)


def payload_bytes(obj) -> int:
    """Bytes of all tensor data inside obj (a tensor, state_dict, dict, list ...)."""
    return int(sum(t.numel() * t.element_size() for t in _iter_tensors(obj)))


def to_bytes(obj) -> bytes:
    """Serialize obj exactly as it will be sent between processes later."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def from_bytes(data: bytes):
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def wire_bytes(obj) -> int:
    return len(to_bytes(obj))


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> tuple[float, float]:
    """Return (average cross-entropy loss, accuracy) of model on loader."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(dim=1) == y).sum().item()
        n += y.numel()
    return total_loss / n, correct / n


# ---------------------------------------------------------------- timing


class Timer:
    """with Timer() as t: ...  then t.seconds"""

    def __enter__(self):
        self._t0 = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        return False


# ---------------------------------------------------------------- CSV logging

ROUNDS_COLUMNS = [
    "run_id", "round", "test_acc", "test_loss",
    "upload_bytes_total", "download_bytes_total",
    "round_wall_s", "server_wait_s", "cum_wall_s", "max_client_compute_s",
]
ROUNDS_COLUMNS_OSFED = ROUNDS_COLUMNS + ["match_loss_mean", "cos_sim_mean", "residual_ratio_mean"]

CLIENTS_COLUMNS = [
    "run_id", "round", "client_id", "n_samples",
    "local_train_s", "snapshot_opt_s", "upload_bytes", "download_bytes",
]
CLIENTS_COLUMNS_OSFED = CLIENTS_COLUMNS + ["match_loss_final", "cos_sim", "residual_ratio"]


class CsvLogger:
    """Append-only CSV writer. Each row is flushed at once, so a stopped run keeps its log."""

    def __init__(self, path: str | Path, columns: Iterable[str]):
        self.path = Path(path)
        self.columns = list(columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        if exists:
            with open(self.path, newline="") as f:
                header = next(csv.reader(f), [])
            if header != self.columns:
                raise ValueError(f"{self.path} already exists with different columns: {header}")
        self._f = open(self.path, "a", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.columns, extrasaction="raise")
        if not exists:
            self._w.writeheader()
            self._f.flush()

    def log(self, **row) -> None:
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
