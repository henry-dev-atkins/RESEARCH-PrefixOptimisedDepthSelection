# Development Environment

## Policy

`uv` is the primary environment and dependency manager for this repository.

The source of truth is:

- `pyproject.toml` for dependency declarations
- `uv.lock` for the resolved environment

Stage 1 does not use `environment.yml`. The spec still mentions it, but the implemented repository currently standardises on `pyproject.toml` plus `uv.lock`.

## Setup

Create or refresh the locked environment with:

```bash
uv lock
uv sync --extra dev
```

`uv` creates a project-local `.venv/`. That directory is ignored by git.

PODS ML runs are GPU-only. The project pins `torch` and `torchvision` to the official PyTorch CUDA wheel index through `pyproject.toml`, so `uv sync` should produce a CUDA-backed environment on supported Windows NVIDIA machines.

Before running any experiment, verify the runtime explicitly:

```bash
uv run python -c "import torch; print({'torch_version': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda, 'device_count': torch.cuda.device_count(), 'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
```

Required fields for a valid ML environment:

- `cuda_available` must be `True`
- `cuda_version` must be non-null
- `device_count` must be at least `1`
- `device_name` must resolve the local NVIDIA GPU

If that check fails, do not run experiments. Repair the environment first.

## Common Commands

Run tests:

```bash
uv run pytest -q
```

Run the MNIST sanity experiment:

```bash
uv run python -m pods.run --config pods/configs/sanity/mnist_mlp.yaml
```

Run the CIFAR-10 sanity experiment:

```bash
uv run python -m pods.run --config pods/configs/sanity/cifar10_cnn.yaml
```

## Dependency Management

When dependencies change:

1. Edit `pyproject.toml`.
2. Regenerate the lockfile with `uv lock`.
3. Refresh the environment with `uv sync --extra dev`.
4. Re-run the relevant tests.

Commit `uv.lock` whenever dependency resolution changes.

## Notes

- `uv run ...` is the standard command surface for tests and experiments.
- If `uv sync` fails because of local cache permissions, rerun it in a shell with normal user permissions. That is an environment issue, not a repo configuration issue.
- CPU-only PyTorch is unsupported for PODS ML runs.
