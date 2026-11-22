"""Step 7 unit checks. No CIFAR-10 needed (uses small random data).

    python -m scripts.test_fedavg
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.client import train_local
from src.data import FastLoader
from src.fedavg import aggregate, client_weights, cosine_lr
from src.metrics import payload_bytes
from src.model import build_model, count_parameters
from src.utils import set_seed


def test_weighted_average_hand_example() -> None:
    a = {"w": torch.tensor([1.0, 1.0]), "n": torch.tensor(10)}
    b = {"w": torch.tensor([5.0, 9.0]), "n": torch.tensor(30)}
    w = client_weights([100, 300], "wavg")            # 0.25 and 0.75
    assert w == [0.25, 0.75], w
    out = aggregate([a, b], w)
    assert torch.equal(out["w"], torch.tensor([4.0, 7.0])), out["w"]   # 0.25*1+0.75*5, 0.25*1+0.75*9
    assert int(out["n"]) == 25 and out["n"].dtype == torch.int64        # integer tensors: rounded average
    out_u = aggregate([a, b], client_weights([100, 300], "uavg"))
    assert torch.equal(out_u["w"], torch.tensor([3.0, 5.0])), out_u["w"]
    print("PASS weighted average (hand example): wavg -> [4, 7], uavg -> [3, 5]")


def test_one_client_equals_local_training() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 255, (300, 3, 32, 32), generator=g, dtype=torch.int64).to(torch.uint8)
    y = torch.randint(0, 10, (300,), generator=g)
    dev = torch.device("cpu")
    epochs, lr = 2, 0.01

    set_seed(7)
    init = build_model("cnn")
    init_state = {k: v.clone() for k, v in init.state_dict().items()}

    # (1) the project code path: train_local + aggregate with one client (weight 1)
    loader_a = FastLoader(x, y, batch_size=64, shuffle=True, seed=3)
    state, _, _ = train_local(build_model("cnn"), init_state, loader_a, epochs, lr, 0.0, 0.0, dev)
    fed = aggregate([state], client_weights([300], "wavg"))

    # (2) an independent, hand-written plain training loop on the same data order
    model = build_model("cnn")
    model.load_state_dict(init_state)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
    loader_b = FastLoader(x, y, batch_size=64, shuffle=True, seed=3)
    for _ in range(epochs):
        for xb, yb in loader_b:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    ref = model.state_dict()
    max_diff = max((fed[k] - ref[k]).abs().max().item() for k in ref)
    assert max_diff < 1e-6, f"1-client FedAvg differs from local training by {max_diff}"
    print(f"PASS 1-client FedAvg == ordinary local training (max abs diff {max_diff:.2e})")


def test_message_bytes_match_model_size() -> None:
    model = build_model("cnn")
    expected = count_parameters(model) * 4            # float32, the CNN has no buffers
    got = payload_bytes({"state": model.state_dict()})
    assert got == expected, (got, expected)
    print(f"PASS upload message = {got:,} bytes = {count_parameters(model):,} parameters x 4 bytes")


def test_cosine_schedule() -> None:
    lrs = [cosine_lr(0.01, r, 100) for r in range(100)]
    assert lrs[0] == 0.01 and all(a > b for a, b in zip(lrs, lrs[1:])) and lrs[-1] > 0
    assert abs(cosine_lr(0.01, 50, 100) - 0.005) < 1e-12
    assert cosine_lr(0.01, 0, 1) == 0.01
    print(f"PASS cosine lr: first {lrs[0]:.5f}, middle {lrs[50]:.5f}, last {lrs[-1]:.7f}")


if __name__ == "__main__":
    test_weighted_average_hand_example()
    test_one_client_equals_local_training()
    test_message_bytes_match_model_size()
    test_cosine_schedule()
    print("\nALL UNIT CHECKS PASSED")
