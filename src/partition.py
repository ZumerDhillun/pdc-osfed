"""Client data partitions (Step 4).

Run from the project root:
    python -m src.partition                      # builds IID and Dirichlet(0.5), 10 clients, seed 100
    python -m src.partition --clients 10 --alpha 0.5 --seed 100 --min-size 100

Outputs:
    data/partitions/cifar10_iid_c10_seed100.json
    data/partitions/cifar10_dir0.5_c10_seed100.json
    results/E0_sanity/partition_cifar10_{iid|dir0.5}_c10_seed100.csv / .png
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.utils import project_root

Parts = Dict[int, np.ndarray]


def iid_partition(n_samples: int, n_clients: int, seed: int) -> Tuple[Parts, int]:
    """Shuffle all indices and cut them into n_clients (almost) equal parts."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    parts = {c: np.sort(p).astype(np.int64) for c, p in enumerate(np.array_split(perm, n_clients))}
    return parts, 1


def dirichlet_partition(
    labels: np.ndarray, n_clients: int, alpha: float, seed: int,
    min_size: int = 100, max_tries: int = 1000,
) -> Tuple[Parts, int]:
    """Per class, draw client proportions from Dirichlet(alpha) and split that class accordingly.

    The whole draw is repeated until every client has at least min_size samples (decision D15).
    Returns (parts, number_of_attempts).
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    for attempt in range(1, max_tries + 1):
        buckets = [[] for _ in range(n_clients)]
        for c in classes:
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            props = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for client_id, piece in enumerate(np.split(idx, cuts)):
                buckets[client_id].extend(piece.tolist())
        if min(len(b) for b in buckets) >= min_size:
            parts = {c: np.sort(np.array(b, dtype=np.int64)) for c, b in enumerate(buckets)}
            return parts, attempt
    raise RuntimeError(
        f"No Dirichlet split with min_size={min_size} after {max_tries} tries. "
        "Lower min_size or change alpha (and write the change in notes/decisions.md)."
    )


def verify_partition(parts: Parts, n_total: int) -> None:
    """Every index 0..n_total-1 must appear exactly once."""
    all_idx = np.concatenate(list(parts.values()))
    assert len(all_idx) == n_total, f"{len(all_idx)} indices, expected {n_total}"
    assert len(np.unique(all_idx)) == n_total, "an index appears more than once"
    assert all_idx.min() == 0 and all_idx.max() == n_total - 1, "index out of range"


def fingerprint(parts: Parts) -> str:
    h = hashlib.sha256()
    for c in sorted(parts):
        h.update(np.asarray(parts[c], dtype=np.int64).tobytes())
    return h.hexdigest()


def partition_name(kind: str, n_clients: int, seed: int, alpha: float = 0.5) -> str:
    tag = "iid" if kind == "iid" else f"dir{alpha}"
    return f"cifar10_{tag}_c{n_clients}_seed{seed}"


def save_partition(path: Path, parts: Parts, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "clients": {str(c): idx.tolist() for c, idx in parts.items()}}
    with open(path, "w") as f:
        json.dump(payload, f)  # no timestamp inside, so re-running gives an identical file


def load_partition(path: Path) -> Tuple[Parts, dict]:
    with open(path) as f:
        payload = json.load(f)
    parts = {int(c): np.asarray(v, dtype=np.int64) for c, v in payload["clients"].items()}
    return parts, payload["meta"]


def class_histogram(labels: np.ndarray, parts: Parts, n_classes: int) -> np.ndarray:
    hist = np.zeros((len(parts), n_classes), dtype=np.int64)
    for c, idx in parts.items():
        hist[c] = np.bincount(np.asarray(labels)[idx], minlength=n_classes)
    return hist


def save_histogram_csv(path: Path, hist: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["client_id", "n_samples"] + [f"class_{k}" for k in range(hist.shape[1])])
        for c in range(hist.shape[0]):
            w.writerow([c, int(hist[c].sum())] + hist[c].tolist())


def save_histogram_png(path: Path, hist: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    n_clients, n_classes = hist.shape
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(n_clients)
    cmap = plt.get_cmap("tab10")
    for k in range(n_classes):
        ax.bar(range(n_clients), hist[:, k], bottom=bottom, color=cmap(k), label=f"class {k}")
        bottom += hist[:, k]
    ax.set_xlabel("client id")
    ax.set_ylabel("number of samples")
    ax.set_title(title)
    ax.set_xticks(range(n_clients))
    ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--min-size", type=int, default=100)
    ap.add_argument("--data-dir", type=str, default=None)
    args = ap.parse_args()

    from src.data import NUM_CLASSES, load_cifar10

    data = load_cifar10(args.data_dir)
    labels = data.y_train.numpy()
    n_total = len(labels)
    root = project_root()
    part_dir = root / "data" / "partitions"
    res_dir = root / "results" / "E0_sanity"

    builds = {
        "iid": lambda: iid_partition(n_total, args.clients, args.seed),
        "noniid": lambda: dirichlet_partition(labels, args.clients, args.alpha, args.seed, args.min_size),
    }
    for kind, build in builds.items():
        parts, attempts = build()
        parts_again, _ = build()  # determinism check: same seed must give the same split
        assert fingerprint(parts) == fingerprint(parts_again), "partition is not reproducible"
        verify_partition(parts, n_total)

        name = partition_name("iid" if kind == "iid" else "dir", args.clients, args.seed, args.alpha)
        meta = {
            "dataset": "cifar10", "type": "iid" if kind == "iid" else "dirichlet",
            "alpha": None if kind == "iid" else args.alpha, "n_clients": args.clients,
            "seed": args.seed, "min_size": args.min_size, "attempts": attempts,
            "n_samples": n_total, "fingerprint": fingerprint(parts),
        }
        save_partition(part_dir / f"{name}.json", parts, meta)
        hist = class_histogram(labels, parts, NUM_CLASSES)
        save_histogram_csv(res_dir / f"partition_{name}.csv", hist)
        save_histogram_png(res_dir / f"partition_{name}.png", hist, name)

        sizes = hist.sum(axis=1)
        print(f"\n{name}: attempts={attempts}  total={int(sizes.sum())}  "
              f"min={int(sizes.min())}  max={int(sizes.max())}")
        print("client sizes:", sizes.tolist())
        print("fingerprint :", meta["fingerprint"][:16], "...")
    print("\nOK: partitions saved in", part_dir)


if __name__ == "__main__":
    main()
