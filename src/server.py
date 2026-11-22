"""Server side (Step 7): holds the global model and evaluates it on the 10,000 test images."""
from __future__ import annotations

import torch

from src.client import State, clone_state
from src.metrics import evaluate


class Server:
    def __init__(self, model: torch.nn.Module, test_loader, device: torch.device):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device

    def get_state(self) -> State:
        """CPU copy of the global weights (the 'download' message)."""
        return clone_state(self.model)

    def set_state(self, state: State) -> None:
        self.model.load_state_dict(state)

    def evaluate(self) -> tuple[float, float]:
        """(test_loss, test_acc) of the global model."""
        return evaluate(self.model, self.test_loader, self.device)
