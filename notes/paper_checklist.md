# Paper checklist: OS-Fed (CVPR 2026). Verified from the paper text. Section and equation numbers are the paper's.

## Method, in the paper's order (Section 3, Figure 2)
1. Client downloads the global model w_t.
2. Local training for K epochs. Accumulated update: Δw = η_local × (sum of the gradients along the local path).
3. Target: Δg = Δw + e. The error e starts at zero and is kept between rounds (Eq. 4).
4. Snapshot S = one grid-like feature map + M² learnable label embeddings. The multi-grid function F cuts the feature map into an M×M grid, resizes each patch to the model input size with bilinear interpolation, pairs it with its label: M² synthetic samples.
5. Tune S so the gradient from those samples on w_t matches Δg. Loss = ||grad − Δg||² / (||Δg||² + ε) (Eq. 5). Optimizer: L-BFGS.
6. New error: e = Δg − (gradient from the final snapshot) (Eq. 6).
7. Client sends only the snapshot (image + labels).
8. Server: join all snapshots along the batch dimension, mean gradient on w_t in one forward-backward pass, w_{t+1} = w_t − that gradient (Eq. 3). No retraining.
9. All clients count equally (Eq. 1 and Eq. 3 use 1/C).

## Settings stated (Section 4.1)
- SGD, initial lr 0.01, cosine schedule; 10 clients; 100 rounds; K = 5 local epochs; Dirichlet alpha 0.5.
- CIFAR: ResNet-18, default grid 4×4. MNIST/FMNIST: MnistNet and MLP, grid 2×2.

## Paper's CIFAR results (Table 2), NOT ours
| Method | CIFAR-10 acc | MB per round | CIFAR-100 acc | MB per round |
|---|---|---|---|---|
| FedAvg | 47.6% | 447.2 | 27.4% | 451.2 |
| OS-Fed 2×2 | 42.2% | 0.12 | 26.9% | 0.12 |
| OS-Fed 4×4 | 44.9% | 0.12 | 30.3% | 0.12 |

## Communication counting
- Table 1: 80.4 MB × 10 clients × 10 rounds = 8,040 MB, 0.57 MB × 10 × 10 = 57 MB. The paper counts uploads only.
- Table 2: 0.12 MB ≈ 10 × 3×32×32 float32 values (12,288 bytes). This shape is our inference, not stated.

## Not stated in the paper text: see notes/decisions.md (D1 to D16)
