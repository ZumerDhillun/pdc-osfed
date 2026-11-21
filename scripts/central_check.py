"""Step 6: centralized sanity check (NOT a federated result).

Run from the project root:
    python -m scripts.central_check --subset 5000 --epochs 2     # quick smoke test
    python -m scripts.central_check --epochs 3                    # the real check (CNN, full data)

Plain SGD, learning rate 0.01, momentum 0, weight decay 0 (decision D2).
Writes results/E0_sanity/central_<model>.csv  (or central_<model>_subset<N>.csv).
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from src.data import FastLoader, load_cifar10
from src.metrics import CsvLogger, evaluate
from src.model import build_model
from src.utils import get_device, project_root, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn", choices=["cnn", "resnet18"])
    ap.add_argument("--norm", default="batch", choices=["batch", "group"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)   # decision D1
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--subset", type=int, default=0, help="use only N training images (0 = all)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("device:", device)
    data = load_cifar10()

    idx = None
    if args.subset > 0:
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(data.y_train), generator=g)[: args.subset].numpy()
    train_loader = FastLoader(data.x_train, data.y_train, indices=idx, batch_size=args.batch_size,
                              shuffle=True, seed=args.seed, device=device)
    test_loader = FastLoader(data.x_test, data.y_test, batch_size=500, shuffle=False, device=device)

    model = build_model(args.model, 10, args.norm).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.0, weight_decay=0.0)

    suffix = f"_subset{args.subset}" if args.subset > 0 else ""
    out_path = project_root() / "results" / "E0_sanity" / f"central_{args.model}{suffix}.csv"
    if out_path.exists():
        out_path.unlink()  # a sanity check is re-run from scratch
    log = CsvLogger(out_path, ["epoch", "train_loss", "test_loss", "test_acc", "epoch_seconds"])

    final_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * y.numel()
            n += y.numel()
        secs = time.perf_counter() - t0
        test_loss, test_acc = evaluate(model, test_loader, device)
        final_acc = test_acc
        log.log(epoch=epoch, train_loss=round(total / n, 4), test_loss=round(test_loss, 4),
                test_acc=round(test_acc, 4), epoch_seconds=round(secs, 1))
        print(f"epoch {epoch}: train_loss={total/n:.4f}  test_loss={test_loss:.4f}  "
              f"test_acc={test_acc:.4f}  ({secs:.1f}s)")
    log.close()
    print("saved", out_path)
    print("CHECK PASSED" if final_acc > 0.15 else "CHECK FAILED: accuracy is not clearly above 10 percent")


if __name__ == "__main__":
    main()
