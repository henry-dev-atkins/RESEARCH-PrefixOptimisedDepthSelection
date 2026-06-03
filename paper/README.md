# Paper Package

This directory holds the manuscript and generated assets for the PODS submission package.

## Generate Assets

```bash
uv run python -m pods.eval.review_figures
uv run python -m pods.eval.export_paper_assets
```

Generated outputs land under:

- `paper/generated/figures/`
- `paper/generated/tables/`

## Build Notes

The manuscript source is expected to live in `paper/main.tex` with section files under `paper/sections/`.
The exact LaTeX build command depends on the local toolchain and should be documented when the manuscript body is added.
