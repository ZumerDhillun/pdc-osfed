"""Step 5 check: sizes, forward pass, one training step, and double-backward test.

Run from the project root:   python -m scripts.check_models
Writes results/E0_sanity/model_sizes.csv. Copy the printed numbers into notes/decisions.md (D9, D12).

The double-backward test mimics what OS-Fed needs later (Eq. 5): the gradient of the loss
with respect to the weights must itself be differentiable with respect to the input image.
"""
from __future__ import annotations

import csv
import time

import torch
import torch.nn.functional as F

from src.metrics import payload_bytes, wire_bytes
from src.model import build_model, count_parameters
from src.utils import get_device, project_root, set_seed


def double_backward_ok(model: torch.nn.Module, device: torch.device) -> bool:
    model.train()
    x = torch.randn(4, 3, 32, 32, device=device, requires_grad=True)
    y = torch.randint(0, 10, (4,), device=device)
    loss = F.cross_entropy(model(x), y)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat = torch.cat([g.reshape(-1) for g in grads])
    target = torch.randn_like(flat)
    match = (flat - target).pow(2).sum() / (target.pow(2).sum() + 1e-8)
    match.backward()
    return x.grad is not None and bool(torch.isfinite(x.grad).all())


def main() -> None:
    set_seed(1)
    device = get_device()
    print("device:", device)
    configs = [("cnn", "batch"), ("resnet18", "batch"), ("resnet18", "group")]
    rows = []
    for name, norm in configs:
        label = name if name == "cnn" else f"{name}-{norm}"
        model = build_model(name, 10, norm).to(device)
        n_params = count_parameters(model)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        p_bytes, w_bytes = payload_bytes(state), wire_bytes(state)

        model.train()
        x = torch.randn(64, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (64,), device=device)
        out = model(x)
        assert out.shape == (64, 10), f"bad output shape {tuple(out.shape)}"
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        t0 = time.perf_counter()
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
        step_s = time.perf_counter() - t0

        dbl = double_backward_ok(build_model(name, 10, norm).to(device), device)
        rows.append({
            "model": label, "parameters": n_params, "payload_bytes": p_bytes,
            "payload_MB": round(p_bytes / 1e6, 3), "wire_bytes": w_bytes,
            "train_step_64_s": round(step_s, 3), "double_backward_ok": dbl,
        })
        print(f"{label:16s} params={n_params:>10,d}  payload={p_bytes/1e6:8.3f} MB  "
              f"wire={w_bytes/1e6:8.3f} MB  step(64)={step_s:5.2f}s  double_backward={dbl}")
        assert dbl, f"double backward failed for {label}"

    out_path = project_root() / "results" / "E0_sanity" / "model_sizes.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nOK: saved", out_path)
    print("Reference: the paper's FedAvg message of 44.7 MB per client implies about 11.2 million float32 parameters.")


if __name__ == "__main__":
    main()
