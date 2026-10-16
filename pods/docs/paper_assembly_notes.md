# Paper Assembly Notes

This document captures the current paper-facing interpretation surface while the repo transitions from the accepted BERT-first Milestone 4 closure toward full-spec completion.

## Scope Lock

Current paper assembly scope:

- accepted BERT-first six-task evidence base
- CIFAR-10 support evidence
- CIFAR-100 deferred from the current package
- new in-progress framework support for:
  - `stsb`
  - `mnli`
  - `progressive_shrinking`
  - `distilled_shallow`
  - broader RoBERTa closure

This is now a transition step between framework hardening and full evidence closure.

## Core Claim Surface

The paper-safe claim is narrow:

- PODS can produce multiple deployable prefixes from one training run.
- Prefix selection can be driven by held-out quality and measured deployment cost.
- The resulting empirical frontier is mixed rather than uniformly dominant.

The paper should not claim:

- PODS broadly beats naive truncation.
- PODS broadly beats independently trained shallow specialists.
- PODS reliably produces materially shallower selected prefixes on every task.

## Current Evidence Story

### What is strong

- The repo has complete BERT-first Milestone 4 evidence across six GLUE tasks.
- Processed summaries cover main variants and the intended static baselines.
- Review figures can now be generated directly from processed summaries.
- Paper summary tables can now be exported from processed summaries.

### What is mixed

- `naive_truncation` remains competitive and is sometimes stronger.
- `qnli` currently leaves the main variants selecting full depth `12`.
- New-task deployment wins are selective rather than broad.

### What is still useful

- `qqp:bert:pods_all` selects depth `8`.
- `cola:bert:pods_all` selects depth `10`.
- The one-run-to-many-prefix deployment story remains materially true even when shallow specialists or truncation win on individual points.

### What is still missing before a full-spec paper claim

- real `stsb` evidence
- real `mnli` evidence
- completed `progressive_shrinking` evidence
- completed `distilled_shallow` evidence
- broader RoBERTa comparator evidence
- full manuscript body and generated submission package

## Figure Workflow

Generate the tracked review figure pack with:

```bash
uv run python -m pods.eval.review_figures
```

Generate the paper package with SVG and PDF vectors and 300 DPI PNG rasters:

```bash
uv run python -m pods.eval.review_figures --output-root paper/figures --datasets cifar10,sst2,mrpc,rte,qqp,qnli,cola,stsb,mnli --formats png,svg,pdf --dpi 300
uv run python -m pods.eval.export_paper_assets --output-root paper --datasets cifar10,sst2,mrpc,rte,qqp,qnli,cola,stsb,mnli
```

The paper exporter fails when a required paper dataset or main variant is absent. GLUE final scores are labelled validation scores because the labelled GLUE evaluation data is the validation split. MNLI reports matched and mismatched validation separately.

Default output root:

- `pods/results/processed/review_figures/`

Useful optional command:

```bash
uv run python -m pods.eval.review_figures --datasets qqp,qnli,cola --formats png
```

Export manuscript tables from processed summaries with:

```bash
uv run python -m pods.eval.export_paper_assets
```

Diff tracked GLUE configs against completed raw runs with:

```bash
uv run python -m pods.experiments.matrix_status
```

## Recommended Figure Shortlist

For a first manuscript pass, prioritise:

1. `milestone4/m4_main_variants_panel`
2. `milestone4/m4_baseline_comparison`
3. `milestone4/selected_depths`
4. `tradeoffs/pareto_core_comparators_latency_new_tasks`
5. `tradeoffs/depth_vs_metric_glue`
6. `overview/pods_flow_diagram`

## Draft Section Guidance

### Introduction

- Emphasise static deployment simplicity.
- Contrast fixed selected prefixes with dynamic early exit complexity.

### Results

- Lead with measured frontier behaviour, not with theoretical layer reduction.
- State openly where full-depth selection persists.
- Keep `naive_truncation` as a first-class comparator.

### Discussion

- Separate evidence-complete from claim-complete.
- Explain that the main value may be deployment frontier coverage from one run, not domination at each depth.

### Limitations

- single-GPU scope
- BERT-first evidence scope
- deferred RoBERTa work
- mixed selected-depth outcomes
- hardware-specific latency

## Immediate Writing Next Steps

1. Draft the abstract and introduction around the static-prefix framing.
2. Lead the results section with the nine-dataset core-variant panel.
3. Use measured frontiers and selected depths to explain deployment trade-offs.
4. Write the discussion and limitations with the mixed result story stated directly.
5. Keep CIFAR-100 and incomplete comparator coverage outside the default package.
