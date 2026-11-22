# Run log: add one line BEFORE every run, including failed runs

| run_id | date | who | purpose | status | notes |
|---|---|---|---|---|---|
| (example) 20261207_A_fedavg_cnn_noniid_smoke_s1__net0 | 2026-12-07 | name | smoke test FedAvg | done | 2 clients, 1 round |
| 20261122_A_fedavg_cnn_noniid_smoke_s1__net0 | 2026-11-22 | oop | smoke test FedAvg (2 clients, 500 samples each, 1 round) | started |  |
| 20261122_A_fedavg_cnn_noniid_smoke_s1__net0 | 2026-11-22 | oop | smoke test FedAvg (2 clients, 500 samples each, 1 round) | done | final test_acc=0.1068 |
| 20261122_A_fedavg_cnn_noniid_r3_s1__net0 | 2026-11-22 | oop | FedAvg 10 clients full data 3 rounds (timing and sanity) | started |  |
| 20261122_A_fedavg_cnn_noniid_r3_s1__net0 | 2026-11-22 | oop | FedAvg 10 clients full data 3 rounds (timing and sanity) | done | final test_acc=0.4127 |
| 20261122_A_fedavg_cnn_noniid_wavg_s1__net0 | 2026-11-22 | oop | FedAvg main baseline, small CNN, non-IID, data-size weighted, seed 1 | started |  |
| 20261122_A_fedavg_cnn_noniid_r3_s1__net0 | 2026-11-22 |  | FedAvg 10 clients full data 3 rounds (timing and sanity) | started |  |
| 20261122_A_fedavg_cnn_noniid_r3_s1__net0 | 2026-11-22 |  | FedAvg 10 clients full data 3 rounds (timing and sanity) | done | final test_acc=0.4127 |
| 20261122_A_fedavg_cnn_noniid_wavg_s1__net0 | 2026-11-22 |  | FedAvg main baseline, small CNN, non-IID, data-size weighted, seed 1 | started |  |
| 20261122_A_fedavg_cnn_noniid_wavg_s1__net0 | 2026-11-22 |  | FedAvg main baseline, small CNN, non-IID, data-size weighted, seed 1 | done | final test_acc=0.6584 |
