"""Find out why training is slow. Run from the project root:

    python -m scripts.bench_cpu

It prints: CPU info, how fast the data loader is (no model), how fast the model trains for
different thread counts, and what that means for one full round and for 100 rounds.
Uses random data, so it needs no dataset and takes about 1 minute.
"""
from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

from src.data import FastLoader
from src.model import build_model

N_TRAIN, K_EPOCHS, N_CLIENTS_SAMPLES = 50_000, 5, 50_000


def cpu_info() -> None:
    print("cpu_count (os)          :", os.cpu_count())
    print("torch default threads   :", torch.get_num_threads())
    print("mkldnn available        :", torch.backends.mkldnn.is_available())
    try:
        print("load average (1,5,15min):", [round(v, 2) for v in os.getloadavg()])
    except OSError:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read()
        model = next((l.split(":")[1].strip() for l in text.splitlines() if l.startswith("model name")), "?")
        flags = next((l for l in text.splitlines() if l.startswith("flags")), "")
        print("cpu model               :", model)
        print("avx / avx2 / avx512     :", " / ".join(str(f" {x} " in flags + " ") for x in ("avx", "avx2", "avx512f")))
    except OSError:
        pass
    print("OMP_NUM_THREADS         :", os.environ.get("OMP_NUM_THREADS", "(not set)"))


def bench_loader(n: int = 6400, bs: int = 64) -> float:
    x = torch.randint(0, 255, (n, 3, 32, 32), dtype=torch.uint8)
    y = torch.randint(0, 10, (n,))
    loader = FastLoader(x, y, batch_size=bs, shuffle=True, seed=0)
    t0 = time.perf_counter()
    for _ in loader:
        pass
    return n / (time.perf_counter() - t0)


def bench_train(threads: int, steps: int = 30, bs: int = 64) -> float:
    torch.set_num_threads(threads)
    model = build_model("cnn")
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(bs, 3, 32, 32)
    y = torch.randint(0, 10, (bs,))
    for _ in range(3):  # warm-up
        opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
    return steps * bs / (time.perf_counter() - t0)


def main() -> None:
    cpu_info()
    print(f"\ndata loader only        : {bench_loader():,.0f} samples/s  (should be above 5,000)")
    default = torch.get_num_threads()
    counts = sorted({1, 2, 4, default, os.cpu_count() or 1})
    results = {}
    print("\nmodel training (small CNN, batch 64):")
    for t in counts:
        results[t] = bench_train(t)
        print(f"  threads={t:<3d} {results[t]:>8,.0f} samples/s")
    best_t = max(results, key=results.get)
    best = results[best_t]
    per_round_s = N_CLIENTS_SAMPLES * K_EPOCHS / best
    print(f"\nbest: threads={best_t} at {best:,.0f} samples/s")
    print(f"one full round (50,000 samples x 5 epochs) = {per_round_s/60:.1f} min")
    print(f"100 rounds = {100*per_round_s/3600:.1f} hours (plus evaluation and OS-Fed snapshot time later)")
    print("\nYour run (c) showed 83 min per round = about "
          f"{N_CLIENTS_SAMPLES*K_EPOCHS/(83*60):,.0f} samples/s.")
    if best < 800:
        print("This machine is slow for this job. Reduce the workload (see the budget formula) or use Kaggle.")
    if results.get(default, best) < 0.5 * best:
        print(f"The default thread count ({default}) is much slower than threads={best_t}: "
              f"run with  OMP_NUM_THREADS={best_t}  in front of the command.")


if __name__ == "__main__":
    main()
