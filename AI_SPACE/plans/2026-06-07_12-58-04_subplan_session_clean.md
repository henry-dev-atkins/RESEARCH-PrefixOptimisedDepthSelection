# [SUBPLAN] Session Clean

Timestamp: 2026-06-07 12:58:04

## Plan Switch

[PLAN_A]: `AI_SPACE/plans/2026-05-13_15-07-34_mnli_bert_matrix_execution.md`

[PLAN_B]: `[SESSION.CLEAN]` cleanup and status reconciliation.

This is an aside. After `[PLAN_B]` is complete, resume `[PLAN_A]` rather than broadening into unrelated experiments.

## Objective

Minimise output sprawl and improve spec alignment by moving old/test/PoC/incremental artefacts into `archive/`, without deleting files and without touching durable run evidence.

## Cleanup Targets

Archive these because they are temporary verification outputs, copied configs/logs, pytest scratch, or promoted one-off tooling:

- `temp/analysis_figures/`
- `temp/matrix_status_verify/`
- `temp/pytest-basetemp-local/`
- `temp/pytest-local/`
- `temp/pytest-of-Henry/`
- `temp/pytest-review-20260512-1203/`
- `temp/pytest-temp/`
- `temp/review_figures_verify/`
- `temp/spec_impl_verify/`
- `temp/spec_impl_verify_2/`
- `temp/*.yaml`
- `temp/*.log`
- `temp/plot_results_review.py`

Keep these because they are operational caches:

- `temp/data/`
- `temp/.uv-cache/`

Keep these because they are durable evidence/reporting paths:

- `pods/results/raw/`
- `pods/results/processed/`
- `paper/generated/`
- `AI_SPACE/plans/`
- `AI_SPACE/handovers/`

## Planned Archive Destination

Move files into:

- `archive/session_clean_2026-06-07_12-58-04/temp/`

## Plan Review

### Overengineering

No new cleanup scripts are needed. A direct, explicit `Move-Item` operation is enough.

### Technical Debt

Leaving operational caches in `temp/` preserves run reproducibility and avoids needless dataset/cache refetches.

### Redundant Iterations

No training runs are needed for cleanup. Verification should be directory listings, git status, and matrix-status inspection only.

### More Elegant Implementation

The cleanest approach is to archive whole temporary directories and orphan temp files, preserving relative names inside a dated archive folder.

### Dream Framework Alignment

This supports the framework by separating durable evidence from exploratory scratch while keeping all artefacts available for audit.

### Gaps, Ambiguity, Contradictions

`temp/analysis_figures/` duplicates durable generated/review figure paths, so it is safe to archive as an old review artefact. If later required, it can be recovered from `archive/`.

## Acceptance Checks

- No file or folder is deleted.
- `temp/data/` and `temp/.uv-cache/` remain in place.
- Cleanup targets exist under `archive/session_clean_2026-06-07_12-58-04/temp/`.
- `git status --short` shows only intentional AI_SPACE/archive/temp movement.
- Matrix status can still be read after cleanup.

## Assumptions

- Cleanup should not modify `AI_SPACE/SPEC.md` because no `[SPEC EDIT REQUIRED]` flag was provided.
- The existing durable results are the current evidence layer and should not be moved.

## Key Tradeoffs

- Moving temporary generated figures means quick visual review from `temp/` is no longer available, but durable generated outputs remain under tracked reporting paths.
- Keeping caches leaves some large data in `temp/`, but avoids unnecessary network/GPU workflow disruption.

## Unresolved Risks

- Some pytest scratch paths may fail ordinary move operations because of Windows permissions.
- If access-denied appears, retry with an approved or escalated move rather than deleting or force-resetting anything.

## Follow-up Work

- Resume `[PLAN_A]` with `mnli_bert_base_pods_all_seed19` only after cleanup and status reconciliation.
- Update `[SESSION.END]` handover with exact cleanup commands and any failed move paths.

## Executive Summary

Archive old temp verification, pytest, copied config/log, and one-off plotting artefacts into a dated archive folder, keep operational caches and durable evidence in place, then return to the MNLI matrix plan.
