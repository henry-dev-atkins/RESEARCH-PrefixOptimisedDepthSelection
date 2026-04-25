# Prefix Optimised Depth Selection (PODS)

PODS is a research repository for training deep models whose prefixes at multiple depths are independently deployable.

## Environment

This repo uses `uv` as the primary Python environment and dependency manager.

Requirements:

- Python 3.11 or newer
- `uv`

Bootstrap the environment with:

```bash
uv lock
uv sync --extra dev
```

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


