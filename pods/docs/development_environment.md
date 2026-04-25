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
- Torch and torchvision currently resolve correctly through `uv` for this repository layout.
