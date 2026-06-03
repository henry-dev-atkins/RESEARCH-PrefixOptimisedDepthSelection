# STS-B BERT Matrix Completion

## Scope of this pass

- execute the full remaining BERT `stsb` matrix from the agreed chunked run plan
- validate each chunk structurally from `results.json`
- regenerate processed status and analysis outputs after each chunk

## Runs completed in this pass

1. `stsb_bert_base_full_seed11`
2. `stsb_bert_base_full_seed19`
3. `stsb_bert_base_pods_all`
4. `stsb_bert_base_pods_all_seed11`
5. `stsb_bert_base_pods_all_seed19`
6. `stsb_bert_base_pods_sandwich`
7. `stsb_bert_base_pods_sandwich_seed11`
8. `stsb_bert_base_pods_sandwich_seed19`
9. `stsb_bert_base_progressive_shrinking_seed11`
10. `stsb_bert_base_progressive_shrinking_seed19`
11. `stsb_bert_base_naive_truncation`
12. `stsb_bert_base_naive_truncation_seed11`
13. `stsb_bert_base_naive_truncation_seed19`
14. `stsb_bert_base_shallow_d2`
15. `stsb_bert_base_shallow_d4`
16. `stsb_bert_base_distilled_shallow_d2`
17. `stsb_bert_base_distilled_shallow_d4`

## Validation outcome

- every run completed on `cuda`
- multi-depth baselines recorded full `selection_results` and `cost_results` over depths `2, 4, 6, 8, 10, 12`
- shallow and distilled baselines stayed single-depth as intended
- regression outputs recorded `spearman_correlation` with `pearson_correlation` in `aux_metrics`

## Processed state after final regeneration

- `pods/results/processed/planning/glue_matrix_status.json`
  - `num_configs = 252`
  - `num_completed = 100`
  - `num_missing = 152`
  - `num_invalid_raw = 5`

- `pods/results/processed/stsb_frontier_summary.json` now includes:
  - `stsb:bert:full`
  - `stsb:bert:pods_all`
  - `stsb:bert:pods_sandwich`
  - `stsb:bert:progressive_shrinking`
  - `stsb:bert:naive_truncation`
  - `stsb:bert:shallow_independent_d2`
  - `stsb:bert:shallow_independent_d4`
  - `stsb:bert:distilled_shallow_d2`
  - `stsb:bert:distilled_shallow_d4`

## Empirical note

- `stsb` selected depths varied meaningfully across methods and seeds
- `shallow_d2` and `distilled_shallow_d2` were materially weaker than `d4` and the multi-depth methods, so `stsb` is not a task where the most aggressive shallow baseline is competitive

## Next tranche

1. execute the remaining BERT `mnli` matrix
2. regenerate matrix status, paper assets, and review figures after each `mnli` chunk
3. then backfill the older-task BERT progressive-shrinking, distilled, and seed-parity gaps
