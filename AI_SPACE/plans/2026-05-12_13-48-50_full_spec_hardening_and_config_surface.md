# Full-Spec Hardening And Config Surface Progress

## Scope of this pass

- harden the current local framework work before large experiment scheduling
- expand the tracked GLUE config surface toward the full agreed matrix
- improve repo tooling so remaining work is measured from config and validated raw runs

## Implemented

- strengthened `distilled_shallow` teacher loading in `pods/src/pods/experiments/runner.py`
  - requires `config.snapshot.yaml`
  - validates teacher baseline kind, task, backbone, pretrained model, candidate depths, and seed
- strengthened `pods/src/pods/experiments/matrix_status.py`
  - valid completion now requires both `results.json` and `config.snapshot.yaml`
  - invalid raw runs are reported separately instead of being counted as completed
- hardened `pods/src/pods/eval/review_figures.py`
  - supports `progressive_shrinking` and distilled shallow display labels
  - tolerates partial dataset coverage instead of assuming every Milestone 4 dataset exists
- added/updated targeted tests for:
  - matrix invalid-run handling
  - sparse review-figure generation
  - distillation teacher metadata mismatch
- expanded tracked GLUE configs to `252` validated YAML files
  - BERT eight-task surface now includes progressive shrinking and distilled shallow variants
  - RoBERTa anchor surface now includes main-seed variants plus comparator/distilled variants

## Correction made during the pass

- a first bulk generation pass incorrectly produced `LayerDrop` configs for `STS-B`
- those invalid files were moved to `archive/invalid_generated_configs/`

## Verification

- direct config validation passed for all `252` tracked GLUE configs
- direct verification passed for:
  - matrix-status invalid-run detection
  - sparse review-figure generation on `stsb`
  - teacher metadata mismatch failure path
- `python pods/src/pods/experiments/matrix_status.py` regenerated `pods/results/processed/planning/glue_matrix_status.json`
- `pytest` still failed for environment temp-dir permission reasons, not for confirmed behavioural regressions

## New matrix truth

- tracked GLUE configs: `252`
- completed valid raw runs: `79`
- missing config-backed runs: `173`
- invalid raw runs flagged separately: `5`

## Remaining work

1. run the four framework-proof GPU experiments:
   - `stsb_bert_base_full`
   - `mnli_bert_base_full`
   - `stsb_bert_base_progressive_shrinking`
   - `sst2_bert_base_distilled_shallow_d2`
2. regenerate processed summaries, review figures, paper assets, and matrix status after each proof block
3. continue the remaining run matrix by diffing against the updated `glue_matrix_status.json`
4. fill the manuscript body in `paper/sections/`
