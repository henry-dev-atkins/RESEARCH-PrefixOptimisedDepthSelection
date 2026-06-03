# MNLI BERT Matrix Execution

## Scope

- execute the remaining BERT `mnli` matrix in small recoverable chunks
- validate each accepted run directly from `results.json`
- regenerate matrix status, paper assets, and review figures after each successful chunk
- record operational learnings and a reproducible handover at the end

## Run chunks

1. full seed parity
   - `mnli_bert_base_full_seed11`
   - `mnli_bert_base_full_seed19`
2. PODS all
   - `mnli_bert_base_pods_all`
   - `mnli_bert_base_pods_all_seed11`
   - `mnli_bert_base_pods_all_seed19`
3. PODS sandwich
   - `mnli_bert_base_pods_sandwich`
   - `mnli_bert_base_pods_sandwich_seed11`
   - `mnli_bert_base_pods_sandwich_seed19`
4. truncation and LayerDrop
   - `mnli_bert_base_naive_truncation`
   - `mnli_bert_base_naive_truncation_seed11`
   - `mnli_bert_base_naive_truncation_seed19`
   - `mnli_bert_base_layerdrop`
   - `mnli_bert_base_layerdrop_seed11`
   - `mnli_bert_base_layerdrop_seed19`
5. progressive shrinking
   - `mnli_bert_base_progressive_shrinking`
   - `mnli_bert_base_progressive_shrinking_seed11`
   - `mnli_bert_base_progressive_shrinking_seed19`
6. shallow and distilled specialists
   - `mnli_bert_base_shallow_d2`
   - `mnli_bert_base_shallow_d4`
   - `mnli_bert_base_distilled_shallow_d2`
   - `mnli_bert_base_distilled_shallow_d4`

## Acceptance checks

- `runtime.device_type == "cuda"`
- `task.dataset == "mnli"`
- processed `mnli` summary remains loadable after each regeneration
- multi-depth baselines must expose depths `2, 4, 6, 8, 10, 12` in `selection_results` and `cost_results`
- single-depth baselines must remain single-depth as intended
- `layerdrop` must not collapse to depth `12` only

## Regeneration commands

```powershell
$env:PYTHONPATH='pods/src'
python -m pods.experiments.matrix_status
python -m pods.eval.export_paper_assets
python -m pods.eval.review_figures
```

## Executive summary

Complete the remaining `mnli_bert_base_*` BERT runs with frequent validation checkpoints so the matrix closes cleanly without drifting into broader backfill or RoBERTa work.
