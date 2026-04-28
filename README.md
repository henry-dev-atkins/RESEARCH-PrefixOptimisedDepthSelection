# Prefix Optimised Depth Selection (PODS)

PODS is a research repository for training deep models whose prefixes at multiple depths are independently deployable.

## Environment

This repo uses `uv` as the primary Python environment and dependency manager.

Requirements:

- Python 3.11 or newer
- `uv`
- NVIDIA GPU with CUDA support for ML runs

Bootstrap the environment with:

```bash
uv lock
uv sync --extra dev
```

Verify that the environment is GPU-backed before running any experiment:

```bash
uv run python -c "import torch; print({'torch_version': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda, 'device_count': torch.cuda.device_count(), 'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
```

PODS ML experiments are GPU-only. CPU-only PyTorch environments are unsupported for training and benchmarking.

The canonical dependency files are:

- `pyproject.toml`
- `uv.lock`

Stage 1 does not use `environment.yml`.

## Development

Run the behavioural test suite with:

```bash
uv run pytest -q
```

Run the first sanity experiment with:

```bash
uv run python -m pods.run --config pods/configs/sanity/mnist_mlp.yaml
```

For environment workflow notes and dependency update policy, see [pods/docs/development_environment.md](pods/docs/development_environment.md).

Current work tracking docs:

- [Session 5 handover](pods/docs/session5_handover_2026-04-28.md)
- [Session 4 handover](pods/docs/session4_handover_2026-04-28.md)
- [Session 3 handover](pods/docs/session3_handover_2026-04-27.md)
- [Session 2 handover](pods/docs/session2_handover_2026-04-26.md)
- [Session 2 TODO](pods/docs/session2_todo.md)
- [Milestone validation](pods/docs/milestone_validation.md)
- [AI learnings log](AI_LEARNINGS_LOG.md)

Current position:

- the strict transformer-proof milestone is now complete
- the chosen BERT-first Milestone 4 sweep is now complete
- the six-task BERT main matrix is complete across `sst2`, `mrpc`, `rte`, `qqp`, `qnli`, and `cola`
- the `qqp`, `qnli`, and `cola` baseline block is complete
- a one-off local figure pack now exists under `temp/analysis_figures/` for result review, including frontier, trade-off, and Pareto-style plots
- the current next-step decision is whether to move into paper assembly from the BERT-first evidence base or reopen the deferred RoBERTa comparator work


