"""FedAvg (Step 7): client update, weighted averaging, learning-rate schedule.

Method interface (OS-Fed will implement the same two methods in Step 8):
    client_update(client, global_state, lr) -> (message, info)
    server_update(server, messages, counts) -> None
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch

from src.client import Client, State, train_local


def cosine_lr(lr0: float, round_idx: int, total_rounds: int) -> float:
    """Decision D3: cosine decay, one value per communication round.

    round_idx is 0-based: round 1 uses lr0, the last round uses a small positive value.
    """
    return 0.5 * lr0 * (1.0 + math.cos(math.pi * round_idx / total_rounds))


def client_weights(counts: Sequence[int], mode: str) -> List[float]:
    """wavg = data-size weights (main baseline, D14). uavg = equal weights (check)."""
    if mode == "wavg":
        total = float(sum(counts))
        return [c / total for c in counts]
    if mode == "uavg":
        return [1.0 / len(counts)] * len(counts)
    raise ValueError(f"unknown aggregation '{mode}' (use 'wavg' or 'uavg')")


def aggregate(states: Sequence[State], weights: Sequence[float]) -> State:
    """Weighted average of model states. Weights must sum to 1.

    Float tensors are averaged. Integer tensors (for example BatchNorm num_batches_tracked)
    are averaged and rounded.
    """
    assert len(states) == len(weights) and len(states) > 0
    assert abs(sum(weights) - 1.0) < 1e-6, "weights must sum to 1"
    out: State = {}
    for key, ref in states[0].items():
        if ref.is_floating_point():
            acc = torch.zeros_like(ref)
            for s, w in zip(states, weights):
                acc.add_(s[key], alpha=float(w))
            out[key] = acc
        else:
            stacked = torch.stack([s[key].double() for s in states])
            w = torch.tensor(list(weights), dtype=torch.double).view(-1, *([1] * (stacked.dim() - 1)))
            out[key] = (stacked * w).sum(0).round().to(ref.dtype)
    return out


class FedAvg:
    name = "fedavg"

    def __init__(self, cfg: Dict, work_model: torch.nn.Module, device: torch.device):
        self.cfg = cfg
        self.work_model = work_model  # reusable model object that clients train in turn
        self.device = device

    def client_update(self, client: Client, global_state: State, lr: float):
        state, loss, seconds = train_local(
            self.work_model, global_state, client.loader,
            epochs=int(self.cfg["local_epochs"]), lr=lr,
            momentum=float(self.cfg["momentum"]), weight_decay=float(self.cfg["weight_decay"]),
            device=self.device,
        )
        message = {"state": state}  # the upload message. Sample counts are metadata (not counted).
        info = {"local_train_s": seconds, "snapshot_opt_s": 0.0, "train_loss": loss}
        return message, info

    def server_update(self, server, messages: Sequence[Dict], counts: Sequence[int]) -> None:
        weights = client_weights(counts, self.cfg["aggregation"])
        server.set_state(aggregate([m["state"] for m in messages], weights))
