"""Run one federated experiment in a single process (Step 7: FedAvg).

    python -m src.run_single --config configs/A_fedavg_cnn_noniid_smoke_s1.yaml
    python -m src.run_single --resume 20261207_A_fedavg_cnn_noniid_wavg_s1__net0
    python -m src.run_single --config ... --stop-after 20      # stop after 20 rounds (resume later)

Writes:  logs/<run_id>/{rounds.csv, clients.csv, config_used.yaml, env.txt, stdout.txt, bytes_summary.json}
         checkpoints/<run_id>/{round_NNN.pt, latest.pt}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from src.client import Client
from src.data import FastLoader, load_cifar10
from src.fedavg import FedAvg, cosine_lr
from src.metrics import (CLIENTS_COLUMNS, ROUNDS_COLUMNS, CsvLogger, payload_bytes, wire_bytes)
from src.model import build_model
from src.partition import load_partition
from src.server import Server
from src.utils import env_report, get_device, project_root, set_seed, today_stamp

METHODS = {"fedavg": FedAvg}  # Step 8 adds "osfed"

REQUIRED = ["tier", "method", "model", "partition", "n_clients", "rounds", "local_epochs",
            "batch_size", "lr", "lr_schedule", "momentum", "weight_decay", "seed", "aggregation"]


class Tee:
    """Print to the screen and to logs/<run_id>/stdout.txt at the same time."""

    def __init__(self, path: Path):
        self.f = open(path, "a")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.stdout.flush()
        self.f.flush()

    def close(self):
        self.f.close()


def save_checkpoint(ckpt_dir: Path, round_idx: int, state, cum_wall_s: float, cfg: dict,
                    run_id: str, keep_round_file: bool) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    blob = {"round": round_idx, "model_state": state, "cum_wall_s": cum_wall_s,
            "run_id": run_id, "config": cfg}
    tmp = ckpt_dir / "latest.tmp"
    torch.save(blob, tmp)
    os.replace(tmp, ckpt_dir / "latest.pt")  # atomic: a crash never leaves a half-written file
    if keep_round_file:
        torch.save(blob, ckpt_dir / f"round_{round_idx:03d}.pt")


def append_run_log(root: Path, run_id: str, cfg: dict, status: str, notes: str = "") -> None:
    """notes/run_log.md is append-only: one line when a run starts, one when it ends."""
    path = root / "notes" / "run_log.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    who = os.environ.get("USER", "")
    line = f"| {run_id} | {datetime.now():%Y-%m-%d} | {who} | {cfg.get('purpose', '')} | {status} | {notes} |\n"
    new = not path.exists()
    with open(path, "a") as f:
        if new:
            f.write("| run_id | date | who | purpose | status | notes |\n|---|---|---|---|---|---|\n")
        f.write(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None, help="run_id to continue from latest.pt")
    ap.add_argument("--condition", type=str, default="net0", help="network condition code (Step 10)")
    ap.add_argument("--stop-after", type=int, default=0, help="stop after this many rounds in total")
    args = ap.parse_args()

    root = project_root()
    if args.resume:
        run_id = args.resume
        run_dir = root / "logs" / run_id
        with open(run_dir / "config_used.yaml") as f:
            cfg = yaml.safe_load(f)
    else:
        if not args.config:
            ap.error("give --config (or --resume)")
        cfg_path = Path(args.config)
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        run_id = f"{today_stamp()}_{cfg_path.stem}__{args.condition}"
        run_dir = root / "logs" / run_id
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise SystemExit(f"config is missing keys: {missing}")
    if cfg["method"] not in METHODS:
        raise SystemExit(f"method '{cfg['method']}' is not available yet")

    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        if (run_dir / "rounds.csv").exists():
            raise SystemExit(f"{run_dir} already has results. Use --resume or change --condition.")
        with open(run_dir / "config_used.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    tee = Tee(run_dir / "stdout.txt")
    sys.stdout = tee
    try:
        run(cfg, run_id, run_dir, root, args)
    finally:
        sys.stdout = tee.stdout
        tee.close()


def run(cfg: dict, run_id: str, run_dir: Path, root: Path, args) -> None:
    device = get_device()
    n_clients, rounds, K = int(cfg["n_clients"]), int(cfg["rounds"]), int(cfg["local_epochs"])
    seed = int(cfg["seed"])
    ckpt_dir = root / "checkpoints" / run_id
    ckpt_every = int(cfg.get("ckpt_every", 1))

    with open(run_dir / "env.txt", "a") as f:
        f.write(env_report() + f"run_id: {run_id}\ndevice: {device}\n\n")
    print(f"run_id : {run_id}\ndevice : {device}\npurpose: {cfg.get('purpose', '')}")

    # ---- data: same partition file for every method (decision D18)
    data = load_cifar10()
    parts, meta = load_partition(root / "data" / "partitions" / f"{cfg['partition']}.json")
    if n_clients > len(parts):
        raise SystemExit(f"partition has {len(parts)} clients, config asks for {n_clients}")

    # ---- model: same seed -> same initial weights for FedAvg and OS-Fed
    set_seed(seed)
    server = Server(build_model(cfg["model"], 10, cfg.get("norm", "batch")),
                    FastLoader(data.x_test, data.y_test, batch_size=500, shuffle=False, device=device),
                    device)
    work_model = build_model(cfg["model"], 10, cfg.get("norm", "batch"))
    method = METHODS[cfg["method"]](cfg, work_model, device)

    # ---- resume
    start_round, cum_wall = 0, 0.0
    latest = ckpt_dir / "latest.pt"
    if args.resume:
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        server.set_state(ck["model_state"])
        start_round, cum_wall = int(ck["round"]), float(ck["cum_wall_s"])
        print(f"resumed from round {start_round}")

    # ---- clients (clients 0..n_clients-1 of the partition)
    spc = cfg.get("samples_per_client")
    clients = []
    for cid in range(n_clients):
        idx = parts[cid]
        if spc:
            rng = np.random.default_rng(seed * 1000 + cid)
            idx = np.sort(rng.choice(idx, size=min(int(spc), len(idx)), replace=False))
        loader = FastLoader(data.x_train, data.y_train, indices=idx, batch_size=int(cfg["batch_size"]),
                            shuffle=True, seed=seed * 100000 + cid * 1000, device=device)
        loader.set_epoch(start_round * K)  # keeps the shuffle order identical after a resume
        clients.append(Client(cid, loader))
    print("client sizes:", [c.n_samples for c in clients])

    rounds_log = CsvLogger(run_dir / "rounds.csv", ROUNDS_COLUMNS)
    clients_log = CsvLogger(run_dir / "clients.csv", CLIENTS_COLUMNS)
    if not args.resume:
        append_run_log(root, run_id, cfg, "started")

    expected_msg_bytes = payload_bytes(server.get_state())  # same number Step 5 printed
    accs, first_up_total = [], None
    last_round = start_round
    for r in range(start_round + 1, rounds + 1):
        t_round = time.perf_counter()
        lr = cosine_lr(float(cfg["lr"]), r - 1, rounds) if cfg["lr_schedule"] == "cosine" else float(cfg["lr"])
        global_state = server.get_state()
        down_each = payload_bytes(global_state)

        messages, counts, losses, times, up_total = [], [], [], [], 0
        for c in clients:
            msg, info = method.client_update(c, global_state, lr)
            up = payload_bytes(msg)
            up_total += up
            messages.append(msg)
            counts.append(c.n_samples)
            losses.append(info["train_loss"])
            times.append(info["local_train_s"] + info["snapshot_opt_s"])
            clients_log.log(run_id=run_id, round=r, client_id=c.client_id, n_samples=c.n_samples,
                            local_train_s=round(info["local_train_s"], 3),
                            snapshot_opt_s=round(info["snapshot_opt_s"], 3),
                            upload_bytes=up, download_bytes=down_each)

        if r == start_round + 1 and not (run_dir / "bytes_summary.json").exists():
            summary = {"download_payload_bytes_per_client": down_each,
                       "download_wire_bytes_per_client": wire_bytes(global_state),
                       "upload_payload_bytes_per_client": payload_bytes(messages[0]),
                       "upload_wire_bytes_per_client": wire_bytes(messages[0])}
            (run_dir / "bytes_summary.json").write_text(json.dumps(summary, indent=2))

        method.server_update(server, messages, counts)
        test_loss, test_acc = server.evaluate()
        round_s = time.perf_counter() - t_round
        cum_wall += round_s
        down_total = down_each * len(clients)
        if first_up_total is None:
            first_up_total = up_total
        rounds_log.log(run_id=run_id, round=r, test_acc=round(test_acc, 4), test_loss=round(test_loss, 4),
                       upload_bytes_total=up_total, download_bytes_total=down_total,
                       round_wall_s=round(round_s, 3),
                       server_wait_s=0.0,  # sequential simulation: meaningful only from Step 9 on
                       cum_wall_s=round(cum_wall, 3), max_client_compute_s=round(max(times), 3))
        accs.append(test_acc)
        print(f"[round {r:3d}/{rounds}] lr={lr:.5f} train_loss={np.mean(losses):.4f} "
              f"test_acc={test_acc:.4f} test_loss={test_loss:.4f} up={up_total/1e6:.2f}MB "
              f"down={down_total/1e6:.2f}MB round={round_s:.1f}s total={cum_wall/60:.1f}min")
        if r == start_round + 1 and rounds - r > 0:
            print(f"   estimate for the remaining {rounds - r} rounds: {(rounds - r) * round_s / 60:.0f} min")

        save_checkpoint(ckpt_dir, r, server.get_state(), cum_wall, cfg, run_id,
                        keep_round_file=(r % ckpt_every == 0 or r == rounds))
        last_round = r
        if args.stop_after and r >= args.stop_after:
            print(f"stopped after round {r} as requested. Continue with: --resume {run_id}")
            break
    rounds_log.close()
    clients_log.close()

    # ---- end-of-run checks
    print("\n=== checks ===")
    if first_up_total is not None:
        ok = first_up_total == expected_msg_bytes * n_clients
        print(f"bytes   : logged upload per round = {first_up_total:,} B; "
              f"{n_clients} x model size from Step 5 = {expected_msg_bytes * n_clients:,} B -> "
              f"{'OK' if ok else 'MISMATCH'}")
    if len(accs) > 1:
        print(f"accuracy: round {last_round - len(accs) + 1} = {accs[0]:.4f} -> round {last_round} = {accs[-1]:.4f} "
              f"-> {'rising' if accs[-1] > accs[0] else 'NOT rising, investigate'}")
    if last_round >= rounds:
        append_run_log(root, run_id, cfg, "done", f"final test_acc={accs[-1]:.4f}" if accs else "")
    print("logs       :", run_dir)
    print("checkpoints:", ckpt_dir)


if __name__ == "__main__":
    main()
