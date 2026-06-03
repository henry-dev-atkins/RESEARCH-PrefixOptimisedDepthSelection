# Analysis And Paper Progress Plan

Timestamp: 2026-05-12 11:40:28

## Intent

Turn the local Milestone 4 review plotting path into tracked repo code, then capture the paper-facing interpretation surface without reopening experiment scope.

## Scope

1. Promote the temporary plotting script into `pods/src/` as a reproducible CLI.
2. Use an intentional durable output root under `pods/results/processed/`.
3. Add focused tests for the new analysis helpers.
4. Add paper-facing notes that state the current evidence story, risks, and next writing steps.

## Non-Scope

- No new training runs.
- No RoBERTa comparator reopening.
- No experiment-logic changes.

## Assumptions

- The processed summary layer remains the source of truth for review plots.
- The BERT-first Milestone 4 closure remains the accepted evidence scope.

## Risks

- The figure path could become bloated if every exploratory view is treated as durable.
- Paper notes must not overstate the mixed result set.

## Tradeoffs

- Promoting the plotter now improves reproducibility and removes dependence on `temp/`.
- Keeping the promoted path summary-driven avoids coupling paper review work to raw-run schema details.

## Executive Summary

The next useful work is not more experiments. It is converting the existing review workflow into a proper repo tool and pinning down the exact paper-safe story from the evidence already collected.
