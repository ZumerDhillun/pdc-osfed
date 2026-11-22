"""Client side (Step 7): one client = its id + its own data loader.

train_local() is the plain local training used by FedAvg: K epochs of SGD starting from the
global weights. It is also what OS-Fed will reuse in Step 8 (the accumulated update Δw is
global weights minus the weights returned here).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

State = Dict[str, torch.Tensor]


@dataclass
class Client:
    client_id: int
    loader: object  # FastLoader over this client's partition

    @property
    def n_samples(self) -> int:
        return self.loader.n_samples


def clone_state(model: torch.nn.Module) -> State:
    """Detached CPU copy of the model's state_dict (this is what gets sent as a message)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_local(
    model: torch.nn.Module,
    global_state: State,
    loader,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    device: torch.device,
) -> Tuple[State, float, float]:
    """Train `model` from `global_state` for `epochs` epochs.

    Returns (new_state_on_cpu, mean_train_loss, seconds).
    `model` is a reusable working copy; its weights are overwritten at the start.
    """
    model.load_state_dict(global_state)
    model.to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    total = torch.zeros((), device=device)
    n = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += loss.detach() * y.numel()  # stays on the device, no sync per step
            n += y.numel()
    mean_loss = (total / n).item()
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    return clone_state(model), mean_loss, seconds
