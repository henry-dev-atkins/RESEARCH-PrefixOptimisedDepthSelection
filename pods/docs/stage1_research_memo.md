# Stage 1 Research Memo

## Exact claim

PODS trains one deep model whose prefixes at selected depths are independently deployable. Deployment selects one fixed prefix offline using held-out validation quality and measured deployment cost.

## Exact non-claims

- PODS is not dynamic early exit.
- PODS does not invent multi-depth inference.
- PODS does not claim prefix specialists beat independently trained shallow models at each matched depth unless experiments show that.
- PODS does not use training accuracy for deployment selection.

## Baselines

- Full model
- Independently trained shallow models
- Naive truncation
- All-exit deep supervision
- LayerDrop-style training
- Once-for-All style progressive shrinking depth baseline
- Distillation baseline when explicitly enabled

## Initial datasets

- Machinery stage: MNIST and CIFAR-10
- First serious image stage: CIFAR-10 and CIFAR-100

## Metrics

- Quality: accuracy, loss
- Deployment: latency p50/p95, throughput, peak memory, serialised model size

## Selection protocol

1. Train split is used for optimisation.
2. A seeded selection-validation split is carved from the training set.
3. The held-out test set is used once for final reporting.
4. Selection chooses the cheapest measured prefix satisfying the quality target.

## Negative-result fallback

If PODS does not beat strong static baselines, the contribution becomes a rigorous empirical study of when static prefix deployment fails and when one-run multi-depth training is still operationally useful.
