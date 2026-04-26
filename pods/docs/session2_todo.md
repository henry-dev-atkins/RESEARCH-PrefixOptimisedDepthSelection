# Session 2 TODO

## Goal

Close the strict transformer-proof milestone by finishing:

- BERT baselines and seeds
- RoBERTa baselines and seeds where required
- LayerDrop seed-7 baseline evidence
- Milestone 3 validation notes

## Current Status

### BERT

`sst2`

- `full`: done
- `pods_all`: done
- `pods_sandwich`: done
- `shallow_d2`: done
- `shallow_d4`: done
- `naive_truncation`: pending
- seeds `11` and `19` for `full` / `pods_all` / `pods_sandwich`: pending

`mrpc`

- `full`: done
- `pods_all`: done
- `pods_sandwich`: done
- `shallow_d2`: pending
- `shallow_d4`: pending
- `naive_truncation`: pending
- seeds `11` and `19` for `full` / `pods_all` / `pods_sandwich`: pending

`rte`

- `full`: done
- `pods_all`: done
- `pods_sandwich`: done
- `shallow_d2`: pending
- `shallow_d4`: pending
- `naive_truncation`: pending
- seeds `11` and `19` for `full` / `pods_all` / `pods_sandwich`: pending

### RoBERTa

- code/config scaffold: in progress in working tree
- behavioural test scaffold: in progress in working tree
- real runs: none completed yet

### LayerDrop

- schema/config scaffold: in progress in working tree
- real runs: none completed yet

## Immediate Next Actions

1. Commit current uncommitted transformer-family and LayerDrop scaffold.
2. Check whether `sst2_bert_base_naive_truncation` partially wrote output.
3. Complete missing BERT seed-7 baselines.
4. Run BERT seeds `11` and `19` for `full`, `pods_all`, `pods_sandwich`.
5. Refresh processed summaries and frontier plots.
6. Start RoBERTa seed-7 main-variant runs.
7. Start LayerDrop seed-7 runs only after BERT closure.
