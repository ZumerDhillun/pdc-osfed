# Decisions (fill in as you go; never code something the paper does not state before writing it here)

Status: DONE = decided, PROVISIONAL = decided for now, revisit at the step shown, OPEN = not decided yet.

| #   | Item | Decision | Status | Date | Who | Reason / where it is decided |
|-----|------|----------|--------|------|-----|------------------------------|
| D1  | Batch size for local training | 64 | DONE | 2026-11-21 | | Paper does not state it. Common default. |
| D2  | Momentum, weight decay | momentum 0, weight decay 0 (plain SGD) | DONE | 2026-11-21 | | Paper formula Δw = η Σ g is exact only for plain SGD. |
| D3  | Cosine schedule scope | Federated runs: lr 0.01 decays with cosine once per communication round over all rounds. Centralized check: constant 0.01 | DONE | 2026-11-21 | | Paper says "cosine schedule" but not the scope. |
| D4  | Snapshot initialization | OPEN | OPEN | | | Step 8.10 (tune on one client, one round). Check supplement. |
| D5  | L-BFGS settings | OPEN | OPEN | | | Step 8.10. |
| D6  | ε in Eq. 5 | 1e-8 | PROVISIONAL | 2026-11-21 | | Small positive value. Revisit in Step 8. |
| D7  | Label format | Soft-label vector, one entry per class (length 10), M² of them | PROVISIONAL | 2026-11-21 | | Paper says "learnable label embeddings". Check supplement. |
| D8  | Loss L for the synthetic gradient | Cross-entropy with soft labels, mean over samples, same on client and server | PROVISIONAL | 2026-11-21 | | Paper says "a loss function". Mean reduction needed for Eq. 3. |
| D9  | BatchNorm | Tier A: SmallCNN with no BatchNorm. Tier B: OPEN (batch or group, 8 groups). Both are implemented in src/model.py | PROVISIONAL | | | Decide after reading the supplement. |
| D10 | Preprocessing, augmentation | CIFAR-10 mean/std normalization, no augmentation | DONE | 2026-11-21 | | Paper does not state it. |
| D11 | CIFAR snapshot shape | 3×32×32 | PROVISIONAL | 2026-11-21 | | Inferred from Table 2 (0.12 MB = 10 × 12,288 bytes). Confirm by measuring. |
| D12 | ResNet-18 variant | CIFAR-style: 3×3 first layer, no max-pool. Measured parameters: 11,173,962 (44.73 MB with BatchNorm, 44.70 MB with GroupNorm) | DONE | 2026-11-21 | | Matches the paper's ~44.7 MB per client message. Re-check with your own run of scripts/check_models.py. |
| D13 | Clients per round | All 10 clients every round | DONE | 2026-11-21 | | Paper: C clients per round. |
| D14 | FedAvg weighting | Main: data-size weighted (wavg). Check on small CNN: equal weights (uavg) | DONE | 2026-11-21 | | Proposal says weighted; paper's Eq. 3 is equal-weight. |
| D15 | Minimum client size, Dirichlet split | 100 samples | DONE | 2026-11-21 | | Our choice. |
| D16 | Supplement read | OPEN | OPEN | | | Download from the CVF page, read before Step 8, update this table. |
| D17 | Byte counting rule | Main number: payload bytes (tensor data only). Also log wire bytes (torch.save size). Report upload, download and total separately. Paper convention = upload only | DONE | 2026-11-21 | | src/metrics.py payload_bytes and wire_bytes. |
| D18 | Seeds | Partition seed 100 (fixed). Training seeds 1, 2, 3 | DONE | 2026-11-21 | | Same partition for both methods. |
| D19 | Data storage | CIFAR-10 cached as uint8 tensors in data/raw/cifar10_uint8.pt, normalized per batch | DONE | 2026-11-21 | | Speed and RAM (10 client processes later). |
