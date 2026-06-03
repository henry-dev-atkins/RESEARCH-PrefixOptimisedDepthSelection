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

## 2026-04-27

### New GLUE tasks should be cache-checked before scheduling a large run block

Observed behaviour:

- `qqp` and `cola` ran immediately because they were already cached locally
- `qnli` failed on first attempt because `glue/qnli` was not present in the local Hugging Face cache

What this means:

- a newly supported task can look implementation-complete while still being operationally blocked by missing cached data
- this is especially easy to miss in a restricted network environment

Operational rule for next time:

- before launching a new task family, verify that its dataset exists in local cache or prefetch it intentionally
- treat first-run dataset availability as a separate operational prerequisite from code correctness

### Config checkpoint commits should happen before long evidence runs

Observed behaviour:

- the Milestone 4 GLUE config expansion existed locally for a while before it was checkpointed
- that increased the risk that an interrupted session would leave the next engineer with local-only configuration truth

What this means:

- long GPU work should not begin while critical experiment configs are still uncommitted

Operational rule for next time:

- after creating a new experiment surface, commit the config layer before running the heavy evidence block
- keep run artefacts gitignored, but keep the experiment definitions in committed state

### Sub-agent summaries are useful but not authoritative for experiment validity

Observed behaviour:

- one sub-agent reported `qqp_bert_base_full_seed11` as structurally invalid
- direct inspection of `results.json` showed the run was actually valid

What this means:

- summarising agents can misread nested experiment payloads, especially around `metric_name` and `metric_value`

Operational rule for next time:

- if an agent reports a structurally suspicious result, inspect `results.json` directly before classifying the run as failed
- use agent summaries as triage, not as final truth

## 2026-04-28

### Cached Hugging Face warning traffic can be noisy without invalidating a run

Observed behaviour:

- multiple GLUE runs emitted:
  - `[WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions`
- accepted runs still completed successfully once the datasets were resolved from local cache

What this means:

- this warning alone does not distinguish between a fatal data-access problem and a harmless cache-backed fallback
- run validity still depends on the written artefacts, not the warning text

Operational rule for next time:

- do not reject a run purely because this warning appears
- inspect `results.json` and processed outputs before classifying the run as failed

### Milestone closure must still validate experiment structure directly

Observed behaviour:

- the remaining Milestone 4 closure work was documentation-heavy, but the accepted experiment set still depended on direct structural validation of each completed run

What this means:

- a successful process exit is weaker evidence than a structurally valid result payload

Operational rule for next time:

- after any long experiment block, verify:
  - `runtime.device_type == "cuda"`
  - the expected task metric is present
  - `selection_results` and `cost_results` cover the intended depths
  - baseline-specific fields match the intended evaluation mode

### Evidence-complete is not the same as claim-complete

Observed behaviour:

- Milestone 4 closed cleanly under the chosen BERT-first scope
- the resulting empirical story across `qqp`, `qnli`, and `cola` remained mixed rather than uniformly positive

What this means:

- completing the planned evidence surface does not automatically justify a strong headline claim

Operational rule for next time:

- separate "the run matrix is complete" from "the hypothesis is strongly supported"
- write the closure docs so they preserve that distinction explicitly

### Selected depth matters more than training intent for the deployment story

Observed behaviour:

- some new-task variants trained successfully but still selected full depth `12`
- the practical deployment value was strongest only where the chosen prefix actually became shallower, such as `qqp:bert:pods_all` at depth `8` and `cola:bert:pods_all` at depth `10`

What this means:

- good final quality alone is not enough for the PODS deployment claim

Operational rule for next time:

- evaluate transformer results first through selected-depth behaviour, then through final quality
- treat full-depth selection as a weaker deployment outcome even when the metric is competitive

### Naive truncation must remain a first-class comparator

Observed behaviour:

- `naive_truncation` was competitive or stronger than PODS on multiple new GLUE tasks

What this means:

- future interpretation cannot assume that jointly trained prefixes dominate a simple post-hoc truncation baseline

Operational rule for next time:

- keep `naive_truncation` in the core comparator set for any follow-on sweep
- avoid broad PODS claims unless the comparison against truncation is explicitly shown task by task

### Processed summaries are rich enough for serious trade-off plots without touching raw runs

Observed behaviour:

- the one-off figure pack for review was generated entirely from `pods/results/processed/*_frontier_summary.json`
- those summaries already exposed:
  - per-depth selection metrics
  - per-depth selection costs
  - selected depths
  - deployed final metrics and costs

What this means:

- most interpretation and figure work can be done from the processed layer alone
- this keeps review work faster and avoids accidental coupling to raw-run schema details

Operational rule for next time:

- default to the processed summaries when building review plots, tables, or decision memos
- only drop to raw runs if a required field is truly absent from the processed layer

### Multi-seed selected-depth summaries should not be collapsed to a rounded mean depth

Observed behaviour:

- one temporary cost-quality plot initially failed because a multi-seed variant had selected depths that averaged to a depth not actually present in the aggregated final-cost keys

What this means:

- selected depth is a discrete observed value, not a continuous plotting coordinate
- a rounded arithmetic mean can invent a non-existent deployment point

Operational rule for next time:

- use an actually observed selected depth for representative deployed-cost views
- if multiple selected depths exist, prefer a mode or explicitly show the full set rather than rounding

## 2026-05-12

### Pytest temp and cleanup paths can fail inside this workspace even when the code under test is fine

Observed behaviour:

- `uv run pytest pods/tests/test_review_figures.py -q` failed before useful test reporting because pytest attempted to create or clean temp directories in locations that raised `PermissionError`
- forcing `--basetemp` into `temp/` avoided the first user-profile temp issue, but pytest session cleanup still hit permission problems on repo-local temp paths
- direct `uv run python` verification of the new review-figure helpers and CLI succeeded

What this means:

- this is an execution-environment problem around pytest temp-dir handling, not evidence that `pods.eval.review_figures` is broken
- direct script verification is a valid fallback when pytest itself cannot complete because of temp-dir permissions

Operational rule for next time:

- if pytest fails during temp-dir setup or cleanup, separate that from actual test failures before debugging code
- prefer a direct `uv run python` verification script for the specific new code path if pytest cannot complete in this environment

### `uv run` and module execution can both be blocked by environment quirks, so direct interpreter fallbacks should be prepared

Observed behaviour:

- `uv run python -m pods.experiments.matrix_status` failed because `C:\Users\Henry\AppData\Local\uv\cache` was not writable
- `python -m pods.experiments.matrix_status` then resolved an older package path rather than the repo-local module, while direct script execution with `PYTHONPATH=pods/src` used the intended file

What this means:

- command failure or stale output here can come from environment resolution rather than experiment-code regressions
- repo-local verification sometimes needs explicit interpreter pathing instead of the nominal CLI entrypoint

Operational rule for next time:

- if `uv run` fails under `AppData`, retry with a workspace-local cache such as `UV_CACHE_DIR=temp/.uv-cache`
- if `python -m pods...` appears to ignore repo edits, rerun with `PYTHONPATH=pods/src` or execute the script file directly

### Bulk config generation must filter unsupported task-method combinations before writing files

Observed behaviour:

- the first bulk config pass generated `LayerDrop` configs for `STS-B`
- current schema rejects that combination because `LayerDrop` is restricted to `text_classification`
- those invalid configs were moved into `archive/invalid_generated_configs/`

What this means:

- large config-surface automation is useful, but it can silently produce spec-invalid files if method constraints are not encoded up front

Operational rule for next time:

- validate the method/task compatibility matrix before bulk-writing configs
- when a generated config is invalid, move it into `archive/` rather than deleting it

### New proof-task datasets should be prefetched before the proof-run gate starts

Observed behaviour:

- the first `stsb_bert_base_full` proof attempt failed before training because `glue/stsb` was not cached locally
- `mnli` would have been blocked for the same reason until both datasets were downloaded into `temp/data/huggingface`

What this means:

- the proof-run gate can still be blocked by operational cache state even after framework work is code-complete

Operational rule for next time:

- before starting a new task-type proof gate, prefetch the missing datasets explicitly
- for this repo, treat `temp/data/huggingface` population as a required precondition when a task has never been run locally

### Config snapshots must round-trip through `load_config` because distillation now depends on them

Observed behaviour:

- the first `sst2_bert_base_distilled_shallow_d2` run failed because `load_config` tried to convert `input_shape: null` from a transformer `config.snapshot.yaml` into `tuple(None)`
- the bug only surfaced once teacher validation began reading saved config snapshots instead of just state dicts

What this means:

- snapshot deserialisation is now part of the runtime path for distillation, not just a reporting artefact

Operational rule for next time:

- whenever config snapshots are written from dataclass state, verify that `load_config(config.snapshot.yaml)` works for that model family
- especially check optional fields that are intentionally `null` for transformer models

### Historical BERT teacher checkpoints can legitimately miss pooler weights

Observed behaviour:

- after fixing snapshot loading, the distilled proof run still failed because the saved `sst2_bert_base_full/model.pt` teacher checkpoint did not include `backbone.pooler.dense.{weight,bias}`
- the current `PodsBert` build path instantiates those parameters even though the project heads consume hidden states directly and do not use pooled outputs

What this means:

- strict checkpoint loading would reject otherwise valid historical teacher runs
- the compatibility gap is narrow and should be handled explicitly rather than silently ignored in general

Operational rule for next time:

- allow only the known optional BERT pooler gap when loading historical teacher checkpoints
- keep all other missing or unexpected keys as hard failures

### STS-B is not forgiving to very shallow BERT baselines, even with distillation

Observed behaviour:

- the completed BERT `stsb` matrix selected reasonable depths for the multi-depth methods, but the single-depth `d2` baselines were much weaker
- `stsb_bert_base_shallow_d2` finished with Spearman around `0.23`
- `stsb_bert_base_distilled_shallow_d2` improved only slightly, to around `0.28`
- `d4` was much healthier for both shallow and distilled variants

What this means:

- `stsb` should not be treated like a task where depth-2 specialists are likely to remain competitive
- distillation does not rescue an overly shallow fixed-depth student on this task

Operational rule for next time:

- when planning the paper narrative, treat `stsb` as evidence that the useful deployment frontier can still have a hard lower-depth floor
- do not assume that adding distillation makes `d2` a serious comparator on every GLUE task

### Interrupted run blocks can leave accepted raw runs behind a stale processed status snapshot

Observed behaviour:

- the `mnli_bert_base_pods_all` chunk was interrupted while the supervising wait was active
- direct inspection afterwards showed `mnli_bert_base_pods_all` and `mnli_bert_base_pods_all_seed11` had already completed successfully on disk, while `mnli_bert_base_pods_all_seed19` had not started or had not finished
- `pods/results/processed/planning/glue_matrix_status.json` still reported those completed runs as missing until the regeneration commands were rerun

What this means:

- after an interruption, the raw run layer may be ahead of the processed reporting layer
- treating the last matrix-status snapshot as truth after an interrupted tranche can misstate real progress

Operational rule for next time:

- after any interrupted experiment block, inspect raw run directories first
- if valid new runs exist, rerun matrix status, paper asset export, and review figures before recording progress or planning the next chunk
