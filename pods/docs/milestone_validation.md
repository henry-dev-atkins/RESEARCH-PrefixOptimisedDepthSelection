## Milestone Validation

This document records the current validated state of Milestones 1, 2, 3, and 4 against `SPEC.md`.

## Current Transition Note

The repo now also contains local framework expansion work for:

1. `stsb`
2. `mnli`
3. `text_regression`
4. `progressive_shrinking`
5. `distilled_shallow`
6. tracked paper-table export

That work is not yet counted as a closed milestone in this document.

Why:

1. the new framework paths have been directly verified and syntax-checked locally, but they have not yet been closed through the full experiment matrix
2. the accepted milestone record below still reflects the last evidence-complete validated state
3. this avoids silently overstating completion while the repo transitions from the BERT-first scope toward the full-spec target

### Milestone 1: Machinery

Deliverables:

1. Repository skeleton: present in the tracked source tree under `pods/src/`, `pods/configs/`, `pods/tests/`, and `pods/docs/`.
2. Seed control: implemented in `pods/src/pods/utils/seeds.py`.
3. Split generation: exercised through the tracked config-driven experiment pipeline and the raw result artefacts under `pods/results/raw/`.
4. Prefix abstraction: covered by the prefix extraction and ResNet tests in `pods/tests/test_prefix_extraction.py` and `pods/tests/test_resnet_and_transforms.py`.
5. Benchmark harness: implemented through the runtime reporting and benchmark pipeline, with runtime metadata in `pods/src/pods/experiments/reporting.py`.
6. MLP sanity run: revalidated with:
   - `pods/results/raw/mnist_mlp_sanity_recheck_a/results.json`
   - `pods/results/raw/mnist_mlp_sanity_recheck_b/results.json`

Exit criteria:

1. Extracted prefixes match in-model prefix logits: PASS
   Evidence:
   - `pods/tests/test_prefix_extraction.py`
   - `pods/tests/test_resnet_and_transforms.py`
2. Real latency/memory metrics are recorded: PASS
   Evidence:
   - both MNIST reruns report `runtime.device_type == "cuda"`
   - both MNIST reruns report non-zero `final_costs[*].peak_memory_mb`
3. All logs are reproducible from configs: PASS
   Evidence:
   - both MNIST reruns were launched from config files
   - selected depth, `final_results`, and `selection_results` matched exactly across reruns
   - runtime metadata now records CUDA version and cuDNN determinism settings

Milestone 1 result: PASS

### Milestone 2: CIFAR proof

Deliverables:

1. CIFAR-10 and CIFAR-100 training runs: PASS
   Evidence:
   - CIFAR-10 raw runs under `pods/results/raw/cifar10_*`
   - CIFAR-100 raw runs under `pods/results/raw/cifar100_*`
2. ResNet-style PODS model: PASS
   Evidence:
   - CIFAR configs under `pods/configs/cifar/`
   - PODS runs for `pods_all` and `pods_sandwich` on CIFAR-10 and CIFAR-100
3. Naive truncation baseline: PASS
   Evidence:
   - `pods/results/raw/cifar10_resnet56_naive_truncation/results.json`
   - `pods/results/raw/cifar100_resnet56_naive_truncation/results.json`
4. Independent shallow baseline: PASS
   Evidence:
   - `pods/results/raw/cifar10_resnet56_shallow_stage1/results.json`
   - `pods/results/raw/cifar10_resnet56_shallow_stage2/results.json`
   - `pods/results/raw/cifar100_resnet56_shallow_stage1/results.json`
   - `pods/results/raw/cifar100_resnet56_shallow_stage2/results.json`
5. Measured frontier plots: PASS
   Evidence:
   - `pods/results/processed/cifar10_frontier.png`
   - `pods/results/processed/cifar10_frontier.svg`
   - `pods/results/processed/cifar100_frontier.png`
   - `pods/results/processed/cifar100_frontier.svg`

Exit criteria:

1. PODS beats naive truncation: FAIL
   Evidence:
   - CIFAR-10 passes this comparison:
     - `pods/results/processed/cifar10_frontier_summary.json`
     - `pods_all` final accuracy mean `0.5171` vs naive truncation `0.4929`
     - `pods_sandwich` final accuracy mean `0.5228` vs naive truncation `0.4929`
   - CIFAR-100 does not pass this comparison:
     - `pods/results/processed/cifar100_frontier_summary.json`
     - naive truncation final accuracy `0.1538`
     - `pods_all` final accuracy mean `0.1423`
     - `pods_sandwich` final accuracy mean `0.1423`
   Interpretation:
   - The current evidence does not support the blanket Milestone 2 claim that PODS beats naive truncation.
   Override status:
   - Accepted by explicit user override and reflected in `SPEC.md`.
2. PODS is compared against independent shallow models: PASS
   Evidence:
   - processed summaries now separate:
     - `shallow_independent_d1`
     - `shallow_independent_d2`
     - `pods_all`
     - `pods_sandwich`
   - comparison artefacts:
     - `pods/results/processed/cifar10_frontier_summary.json`
     - `pods/results/processed/cifar100_frontier_summary.json`
     - `pods/results/processed/cifar10_frontier_points.json`
     - `pods/results/processed/cifar100_frontier_points.json`
3. At least three seeds for main variants: PASS
   Evidence:
   - CIFAR-10:
     - `full`: seeds `7, 11, 19`
     - `pods_all`: seeds `7, 11, 19`
     - `pods_sandwich`: seeds `7, 11, 19`
   - CIFAR-100:
     - `full`: seeds `7, 11, 19`
     - `pods_all`: seeds `7, 11, 19`
     - `pods_sandwich`: seeds `7, 11, 19`
   Source:
   - `pods/results/processed/cifar10_frontier_summary.json`
   - `pods/results/processed/cifar100_frontier_summary.json`

Milestone 2 result: PASS BY EXPLICIT OVERRIDE

### Milestone 3: Transformer proof

Deliverables:

1. BERT-base PODS wrapper: PASS
   Evidence:
   - `pods/src/pods/models/bert.py`
   - `pods/src/pods/models/transformer_prefix.py`
   - `pods/tests/test_transformer_prefix_extraction.py`
2. RoBERTa-base PODS wrapper: PASS
   Evidence:
   - `pods/src/pods/models/roberta.py`
   - `pods/src/pods/models/transformer_prefix.py`
   - `pods/tests/test_transformer_prefix_extraction.py`
   - `pods/results/raw/sst2_roberta_base_full/results.json`
3. GLUE splits: PASS
   Evidence:
   - `pods/src/pods/data/datasets.py`
   - supported tracked tasks: `sst2`, `mrpc`, `rte`
4. Shallow BERT baselines: PASS
   Evidence:
   - `pods/results/raw/sst2_bert_base_shallow_d2/results.json`
   - `pods/results/raw/sst2_bert_base_shallow_d4/results.json`
   - `pods/results/raw/mrpc_bert_base_shallow_d2/results.json`
   - `pods/results/raw/mrpc_bert_base_shallow_d4/results.json`
   - `pods/results/raw/rte_bert_base_shallow_d2/results.json`
   - `pods/results/raw/rte_bert_base_shallow_d4/results.json`
5. LayerDrop-style baseline: PASS
   Evidence:
   - `pods/results/raw/sst2_bert_base_layerdrop/results.json`
   - `pods/results/raw/mrpc_bert_base_layerdrop/results.json`
   - `pods/results/raw/rte_bert_base_layerdrop/results.json`
   - processed summaries:
     - `sst2:bert:layerdrop`
     - `mrpc:bert:layerdrop`
     - `rte:bert:layerdrop`

Exit criteria:

1. At least three GLUE tasks complete: PASS
   Evidence:
   - BERT task coverage is complete for `sst2`, `mrpc`, and `rte`
   - main variants `full`, `pods_all`, and `pods_sandwich` are present for all three tasks
2. Extracted prefix benchmarking works: PASS
   Evidence:
   - `pods/src/pods/experiments/runner.py` benchmarks `model.extract_prefix(depth)`
   - extraction parity is covered by `pods/tests/test_transformer_prefix_extraction.py`
   - LayerDrop reruns now record multi-depth `selection_results` and `cost_results` across depths `2, 4, 6, 8, 10, 12`
3. No hidden-state shortcut is used in cost measurement: PASS
   Evidence:
   - `pods/src/pods/experiments/runner.py` calls `benchmark_prefix` on extracted prefix modules
   - `pods/src/pods/eval/benchmark.py` measures the extracted module forward pass directly

Milestone 3 result: PASS

### Session 2 Position After Milestone 3

1. The strict `SPEC.md` Milestone 3 bar is now complete.
2. This is not the same as full Session 2 comparator closure.
3. The remaining Session 2 gap is the incomplete RoBERTa main-variant block beyond the accepted `sst2_roberta_base_full` run.

### Milestone 4: Full empirical sweep under the chosen BERT-first scope

Scope used for this validation pass:

- BERT-first only
- six-task GLUE classification block:
  - `sst2`
  - `mrpc`
  - `rte`
  - `qqp`
  - `qnli`
  - `cola`
- `STS-B` deferred
- `MNLI` deferred
- RoBERTa not required for Milestone 4 closure

Deliverables:

1. Main BERT matrix across three seeds for all six tasks: PASS
   Evidence:
   - processed summaries:
     - `pods/results/processed/sst2_frontier_summary.json`
     - `pods/results/processed/mrpc_frontier_summary.json`
     - `pods/results/processed/rte_frontier_summary.json`
     - `pods/results/processed/qqp_frontier_summary.json`
     - `pods/results/processed/qnli_frontier_summary.json`
     - `pods/results/processed/cola_frontier_summary.json`
   - each task now has:
     - `full`
     - `pods_all`
     - `pods_sandwich`
     - seeds `7, 11, 19`
2. New-task baseline surface for `qqp`, `qnli`, and `cola`: PASS
   Evidence:
   - `qqp` raw baseline runs:
     - `pods/results/raw/qqp_bert_base_shallow_d2/results.json`
     - `pods/results/raw/qqp_bert_base_shallow_d4/results.json`
     - `pods/results/raw/qqp_bert_base_naive_truncation/results.json`
     - `pods/results/raw/qqp_bert_base_layerdrop/results.json`
   - `qnli` raw baseline runs:
     - `pods/results/raw/qnli_bert_base_shallow_d2/results.json`
     - `pods/results/raw/qnli_bert_base_shallow_d4/results.json`
     - `pods/results/raw/qnli_bert_base_naive_truncation/results.json`
     - `pods/results/raw/qnli_bert_base_layerdrop/results.json`
   - `cola` raw baseline runs:
     - `pods/results/raw/cola_bert_base_shallow_d2/results.json`
     - `pods/results/raw/cola_bert_base_shallow_d4/results.json`
     - `pods/results/raw/cola_bert_base_naive_truncation/results.json`
     - `pods/results/raw/cola_bert_base_layerdrop/results.json`
3. Multi-depth deployable-prefix baselines preserve selectable prefix evidence: PASS
   Evidence:
   - `qqp`, `qnli`, and `cola` `naive_truncation` runs all record `selection_results` and `cost_results` across depths `2, 4, 6, 8, 10, 12`
   - `qqp`, `qnli`, and `cola` `layerdrop` runs also record `selection_results` and `cost_results` across depths `2, 4, 6, 8, 10, 12`
   - no new-task LayerDrop run collapsed to deepest-only evaluation

Exit criteria:

1. Main matrix complete for six tasks, three variants, and three seeds: PASS
   Evidence:
   - `54 / 54` BERT main-matrix runs are present and processed under the chosen Milestone 4 scope
2. New tasks are compared against the intended static baselines: PASS
   Evidence:
   - processed summaries for `qqp`, `qnli`, and `cola` each contain:
     - `full`
     - `pods_all`
     - `pods_sandwich`
     - `shallow_independent_d2`
     - `shallow_independent_d4`
     - `naive_truncation`
     - `layerdrop`
3. The processed summaries are sufficient to support the intended Milestone 4 comparisons: PASS
   Evidence:
   - `qqp`
     - `full` mean accuracy `0.9056888449171407`
     - `pods_all` mean accuracy `0.8995135625360706`
     - `pods_sandwich` mean accuracy `0.9059279412977163`
     - `naive_truncation` accuracy `0.9067771456838981`
   - `qnli`
     - `full` mean accuracy `0.9123802550491183`
     - `pods_all` mean accuracy `0.9087192629202514`
     - `pods_sandwich` mean accuracy `0.9117090731588261`
     - `naive_truncation` accuracy `0.9093904448105437`
   - `cola`
     - `full` mean Matthews correlation `0.5507561605669934`
     - `pods_all` mean Matthews correlation `0.5524357115340137`
     - `pods_sandwich` mean Matthews correlation `0.5502251919574714`
     - `naive_truncation` Matthews correlation `0.5609903802347734`
   Interpretation:
   - `qqp` provides only a partial PODS win:
     - `pods_sandwich` roughly matches `full`, but still selects depth `12`
     - `pods_all` selects depth `8` consistently, but loses quality
     - `naive_truncation` is slightly stronger than both PODS variants on final accuracy
   - `qnli` is the weakest new-task PODS result:
     - all three main variants select full depth `12`
     - this leaves no meaningful shallower deployment outcome under the current selection policy
   - `cola` is the most positive new-task result, but still mixed:
     - `pods_all` slightly exceeds `full` while selecting depth `10`
     - `naive_truncation` remains strongest on final Matthews correlation

Milestone 4 result: PASS

### Residual Caveats

1. Milestone 2 was accepted by explicit user override after the spec was updated to allow an objective-mismatch override with supporting analysis.
2. The current CIFAR-100 evidence still does not show PODS beating naive truncation in raw measured accuracy terms and should continue to be analysed rather than ignored.
3. The benchmark component is reproducible enough for milestone validation, but wall-clock latency itself still varies slightly across reruns, so exact benchmark-metric identity should not be expected.
4. Processed summaries and plots are generated from valid CUDA-backed runs only and should be treated as the current source of truth for this milestone pass.
5. Milestone 3 closure here is the strict spec-minimum transformer proof, not a claim that the broader RoBERTa comparator programme is complete.
6. Milestone 4 closure here uses the explicitly chosen BERT-first scope and should not be read as RoBERTa comparator closure.
7. The completed Milestone 4 evidence is mixed rather than uniformly positive:
   - `qqp` favours `pods_sandwich` over `pods_all`, but `naive_truncation` slightly exceeds both PODS variants in final accuracy
   - `qnli` leaves all three main variants selecting full depth `12`
   - `cola` shows `pods_all` slightly ahead of `full`, but `naive_truncation` remains strongest in final Matthews correlation
8. Selected-depth behaviour should be interpreted honestly:
   - `qqp:bert:pods_all` selected depth `8` for all three seeds
   - `cola:bert:pods_all` selected depth `10` for all three seeds
   - most other new-task main variants still selected full depth `12`
9. Milestone 4 is therefore closed as an evidence milestone under the chosen scope, not as proof that PODS uniformly dominates the static baselines on GLUE.
