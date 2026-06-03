# Proof Gate Completion

## Scope of this pass

- execute the four framework-proof runs required before broader matrix expansion
- fix any real framework bugs exposed by those runs
- regenerate processed status and analysis outputs after the proof block

## Executed proof runs

1. `stsb_bert_base_full`
2. `mnli_bert_base_full`
3. `stsb_bert_base_progressive_shrinking`
4. `sst2_bert_base_distilled_shallow_d2`

## Real blockers encountered

### 1. Missing cached datasets

- initial `stsb` proof run failed because `glue/stsb` was not present in local cache
- `stsb` and `mnli` were then fetched into `temp/data/huggingface`

### 2. Distillation snapshot round-trip bug

- `load_config` failed on teacher `config.snapshot.yaml` because transformer snapshots contained `input_shape: null`
- fixed in `pods/src/pods/config/schema.py`

### 3. Distillation teacher checkpoint compatibility bug

- historical `sst2_bert_base_full/model.pt` teacher checkpoint was missing optional BERT pooler weights
- fixed in `pods/src/pods/experiments/runner.py` by allowing only the known pooler gap while keeping other mismatches fatal

## Outcome

All four proof runs now complete successfully.

### Verified proof results

- `stsb_bert_base_full`
  - CUDA-backed
  - regression metrics recorded correctly
  - primary metric: `spearman_correlation`
  - auxiliary metric: `pearson_correlation`

- `mnli_bert_base_full`
  - CUDA-backed
  - primary final result recorded on matched validation
  - `final_results_secondary` recorded on mismatched validation

- `stsb_bert_base_progressive_shrinking`
  - full multi-depth `selection_results`
  - full multi-depth `cost_results`
  - selected depth `10`

- `sst2_bert_base_distilled_shallow_d2`
  - CUDA-backed
  - distillation teacher/student path completed
  - selected depth `2`
  - result schema matches other baselines

## Processed state after regeneration

- `pods/results/processed/planning/glue_matrix_status.json`
  - `num_configs = 252`
  - `num_completed = 83`
  - `num_missing = 169`
  - `num_invalid_raw = 5`

- regenerated successfully:
  - matrix status
  - paper asset export
  - review figures

## Next concrete tranche

1. complete the remaining BERT `stsb` matrix
2. complete the remaining BERT `mnli` matrix
3. then continue the missing progressive-shrinking and distilled runs across the older six-task BERT block
4. defer broader RoBERTa expansion until the BERT `stsb` and `mnli` surfaces are materially closed
