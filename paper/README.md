# PODS paper package

This directory contains paper-ready figures and tables generated from fresh processed experiment summaries.

## Generate the package

Run the figure export first:

```bash
uv run python -m pods.eval.review_figures \
  --processed-root pods/results/processed \
  --output-root paper/figures \
  --datasets cifar10,sst2,mrpc,rte,qqp,qnli,cola,stsb,mnli \
  --formats png,svg,pdf \
  --dpi 300
```

Then export the tables:

```bash
uv run python -m pods.eval.export_paper_assets \
  --processed-root pods/results/processed \
  --output-root paper \
  --datasets cifar10,sst2,mrpc,rte,qqp,qnli,cola,stsb,mnli
```

The current strict paper scope includes CIFAR-10 and the configured GLUE tasks. CIFAR-100 remains deferred and is excluded from the default package. The table command fails if one of the active paper datasets or main variants is missing.

## Outputs

Figures are available as SVG and PDF vectors and as 300 DPI PNG rasters. The figure manifest lists every generated file.

The core tables contain one row for each publication dataset and core variant. The core deployment table records selected depth, latency, memory, and model size. Per-dataset summary and deployment tables remain available for audit and appendix use. MNLI also has a secondary validation table for the mismatched split.

The summary tables report the final task metric mean and standard deviation across completed runs. Selected depths and seed IDs are retained. CSV values retain source precision. The split columns state whether the final score came from an image test split, a GLUE validation split, or an MNLI matched or mismatched validation split.

Paper claims must use the processed summaries and preserve the mixed result pattern. PODS is a static offline prefix-selection method. It is not dynamic early exit.

## Presentation order

1. Introduce static prefix selection from one trained model.
2. Use `overview/pods_flow_diagram` to show training, measurement, selection, and fixed-prefix deployment.
3. Use the main-variant panel to compare full, PODS all, and PODS sandwich across the nine tasks.
4. Use the frontier plots and selected-depth plot to show the quality and deployment-cost trade-off.
5. State mixed outcomes directly. Do not claim uniform dominance over truncation or shallow specialists.
6. Use the CSV and LaTeX tables for the main text and appendix. Treat optional baseline rows as supporting evidence unless their coverage is complete.

CIFAR-100 is deferred from the default package. Historical CIFAR-100 evidence remains outside this publication scope.
