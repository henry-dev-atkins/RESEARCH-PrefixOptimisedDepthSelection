# AI Learnings Log

## 2026-04-26

### `uv` cache access can fail from the sandboxed agent environment

Observed behaviour:

- `uv run ...` initially failed with:
  - `Failed to initialize cache at C:\Users\Henry\AppData\Local\uv\cache`
  - `Access is denied. (os error 5)`

What this means:

- this failure is environmental
- it does not indicate a regression in `pods.run`
- rerunning with `UV_CACHE_DIR` pointed into `temp/.uv-cache` resolved it without changing experiment code

Operational rule for next time:

- if a run fails on `uv` cache initialisation under `AppData`, treat it as an execution-environment issue first
- first retry with `UV_CACHE_DIR=temp/.uv-cache`
- do not start debugging experiment code until that retry has been attempted

### Large experiment blocks should be chunked so interruptions do not lose a whole session

Observed behaviour:

- a single 18-run BERT multi-seed block was interrupted before a completion report was returned
- restarting in 3-run dataset chunks preserved progress and made validation straightforward

Operational rule for next time:

- group long transformer experiment work into small awaiter batches
- validate `results.json` after each chunk
- after any user interruption, inspect raw output directories before assuming a run completed or failed

### LayerDrop evaluation can silently collapse to deepest-only if it is treated like a fixed-depth baseline

Observed behaviour:

- the first accepted `layerdrop` runs completed successfully but recorded `selection_results` only for depth `12`
- the cause was workflow logic in `pods/src/pods/experiments/runner.py` that evaluated only the deepest depth for `baseline.kind == "layerdrop"`

What this means:

- this is a real implementation bug, not an execution-environment issue
- LayerDrop must be evaluated like a multi-depth deployable-prefix baseline after training
- otherwise the run appears valid while failing the intended Milestone 3 evidence standard

Operational rule for next time:

- for any baseline that is meant to produce selectable prefixes, validate that `selection_results` and `cost_results` cover every candidate depth
- do not accept a successful run purely because `results.json` exists and `runtime.device_type == "cuda"`
