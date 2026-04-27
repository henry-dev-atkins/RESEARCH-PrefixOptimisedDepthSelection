## Milestone Validation

This document records the current validation state of Milestones 1, 2, and 3 against `SPEC.md`.

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

### Residual Caveats

1. Milestone 2 was accepted by explicit user override after the spec was updated to allow an objective-mismatch override with supporting analysis.
2. The current CIFAR-100 evidence still does not show PODS beating naive truncation in raw measured accuracy terms and should continue to be analysed rather than ignored.
3. The benchmark component is reproducible enough for milestone validation, but wall-clock latency itself still varies slightly across reruns, so exact benchmark-metric identity should not be expected.
4. Processed summaries and plots are generated from valid CUDA-backed runs only and should be treated as the current source of truth for this milestone pass.
5. Milestone 3 closure here is the strict spec-minimum transformer proof, not a claim that the broader RoBERTa comparator programme is complete.
