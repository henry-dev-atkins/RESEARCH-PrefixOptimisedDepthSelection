# Complete Research Project Specification

**Working method name:** Prefix Optimised Depth Selection, abbreviated **PODS**
**Project type:** empirical machine-learning systems/research project  
**Primary goal:** train one over-depth model whose prefixes are valid deployable models, then statically select the shallowest prefix satisfying a measured validation-quality target and measured deployment-cost constraint.  
**Deployment mode:** static depth selection, not dynamic early exit.  
**Hardware assumption:** one local NVIDIA RTX 5090-class GPU.

---


### Key Properties

- **Single Training Run → Multiple Deployable Models**
- **No Runtime Overhead (no routing / gating)**
- **True Resource Reduction (latency, memory, size)**
- **Deterministic Inference Graph**
- **Compatible with standard deployment stacks (ONNX, TensorRT, etc.)**

---

### Non-Goals

- No per-example dynamic depth selection
- No reliance on full-model agreement or distillation (unless explicitly added)
- No theoretical FLOP-only claims — all evaluation must use real measurements

---

### Future Extensions

- Width elasticity (prefix + channel scaling)
- Expert selection (MoE-style extensions)
- Hybrid static + dynamic routing
- Prefix distillation for further compression


### Summary
## During Training

1. **Model Construction**
   - Build a base model with `L` sequential layers.
   - Define a set of candidate deployment depths:
     ```
     D = {d1, d2, ..., dk} ⊆ {1, ..., L}
     ```
   - Attach a lightweight prediction head `H_d` at each depth `d ∈ D`.

2. **Forward Pass (Multi-Depth)**
   - For each input batch:
     - Run the model sequentially through layers.
     - At each depth `d ∈ D`, compute:
       ```
       ŷ_d = H_d(h_d)
       ```

3. **Loss Computation**
   - Compute task loss at each depth:
     ```
     ℓ_d = Loss(ŷ_d, y)
     ```
   - Combine losses using depth-weighted aggregation:
     ```
     L_total = Σ_d α_d * ℓ_d
     ```
   - Typical weighting:
     ```
     α_d = (d / L)^p
     ```

4. **Depth Sampling (Optional - Sandwich Training)**
   - Instead of computing all depths every step:
     - Always include:
       - shallowest depth `d_min`
       - deepest depth `d_max`
     - Sample 1–2 intermediate depths per batch.
   - This reduces compute and improves training stability.

5. **Backpropagation**
   - Backpropagate through shared layers using the aggregated loss.
   - All prefix heads contribute gradients to early layers.

6. **Validation Tracking**
   - At evaluation intervals:
     - Measure validation performance for each prefix depth:
       ```
       { (d, metric_d, loss_d) }
       ```
   - Store full depth-performance curves.

---

## During Deployment

1. **Prefix Evaluation (Offline)**
   - For each candidate depth `d ∈ D`:
     - Extract the prefix model:
       ```
       M_d = (layers 1 → d) + H_d
       ```
     - Measure:
       - Task quality (accuracy, loss, etc.)
       - Real latency (p50 / p95)
       - Throughput (samples/sec)
       - Peak memory usage
       - Model size on disk

2. **Depth Selection**
   - Define a target quality constraint:
     - Absolute:
       ```
       metric_d ≥ target
       ```
     - or relative to full model:
       ```
       metric_d ≥ metric_L - ε
       ```
   - Select optimal depth:
     ```
     d* = argmin_d Cost(d)
     subject to quality constraint
     ```

3. **Model Extraction**
   - Materialise the selected prefix:
     ```
     Deployable model = layers[1:d*] + H_{d*}
     ```
   - Remove all unused deeper layers.

4. **Deployment**
   - Deploy the extracted prefix as a **fixed, static model**:
     - No dynamic routing
     - No conditional execution
     - Fully compatible with batching and compilation

5. **Inference**
   - Standard forward pass through depth `d*` only:
     ```
     ŷ = M_{d*}(x)
     ```

---



## 1. Research Thesis

The core hypothesis is:

> A single jointly trained deep network can produce multiple deployable prefix subnetworks. After training, engineers can evaluate each prefix on held-out validation data and deploy the shallowest prefix that satisfies a quality/cost target.

Let a model contain \(L\) sequential blocks:

\[
h_0 = E(x)
\]

\[
h_\ell = F_\ell(h_{\ell-1}; \theta_\ell), \qquad \ell = 1, \dots, L
\]

where \(E\) is an input embedding, stem, or feature extractor, and \(F_\ell\) is the \(\ell\)-th model block.

For candidate deployment depths:

\[
\mathcal{D} \subseteq \{1, \dots, L\}
\]

attach a task head at each candidate depth:

\[
\hat{y}_d = H_d(h_d; \phi_d), \qquad d \in \mathcal{D}
\]

The deployable prefix model at depth \(d\) is:

\[
M_d(x) = H_d(F_d \circ F_{d-1} \circ \cdots \circ F_1(E(x)))
\]

After training, choose:

\[
d^* = \arg\min_{d \in \mathcal{D}} \operatorname{Cost}_{measured}(M_d)
\]

subject to a measured validation-quality constraint:

\[
Q_{val}(M_d) \ge Q_{target}
\]

or, for a loss-minimisation objective:

\[
L_{val}(M_d) \le L_{target}
\]

At deployment time, only the selected prefix is retained:

\[
E, F_1, \dots, F_{d^*}, H_{d^*}
\]

The unused later layers:

\[
F_{d^*+1}, \dots, F_L
\]

are discarded.

This is not dynamic early exit. There is no per-example router, no fallback, no inference-time branching, and no batch divergence. The final deployed model is a fixed computation graph.

---

## 2. Non-Negotiable Scientific Standard

The project must not claim:

> We reduce cost because the prefix has fewer theoretical FLOPs.

That is not sufficient for a serious paper.

The paper may claim cost reduction only from **real measurements**:

1. task quality: measured validation/test loss and task metric;
2. latency: measured wall-clock GPU latency;
3. throughput: measured examples/sec or tokens/sec;
4. memory: measured peak allocated GPU memory and model file size;
5. optional energy: measured GPU-reported power integration or external wattmeter measurement.

Estimated FLOPs/MACs may appear only as appendix diagnostics. They should not be the headline evidence.

The project should follow the reproducibility and evidence standards expected by top ML venues. ICML reviewing criteria emphasise clear claims supported by reproducible experiments or sound theory, and NeurIPS requires reproducibility, transparency, ethics, and societal-impact information in its paper checklist.

Relevant guidelines:

- ICML Call for Papers / reviewing expectations: <https://icml.cc/Conferences/2025/CallForPapers>
- NeurIPS paper checklist: <https://neurips.cc/public/guides/PaperChecklist>

---

## 3. Prior-Art Positioning

This work is adjacent to several existing families. The paper must be precise about what is new and what is not.

### 3.1 Early-exit networks

BranchyNet and related early-exit networks attach intermediate classifiers and often use dynamic sample-wise exiting. BranchyNet focuses on exiting early for easy samples.

Reference:

- BranchyNet: <https://arxiv.org/abs/1709.01686>

### 3.2 Early-exit transformers

DeeBERT applies early-exit ideas to BERT/RoBERTa-style models. It attaches off-ramps to transformer layers and uses dynamic early exiting.

References:

- DeeBERT ACL paper: <https://aclanthology.org/2020.acl-main.204/>
- DeeBERT PDF: <https://cs.uwaterloo.ca/~jimmylin/publications/Xin_etal_ACL2020.pdf>

### 3.3 LayerDrop

LayerDrop trains transformer models with structured layer dropout, so subnetworks of different depths can be selected at inference time with limited performance loss. This is a direct baseline.

Reference:

- LayerDrop: <https://arxiv.org/abs/1909.11556>

### 3.4 Once-for-All networks

Once-for-All trains a single network supporting many architectural settings, including depth and width, then selects specialised subnetworks for deployment without retraining.

Reference:

- Once-for-All: <https://arxiv.org/abs/1908.09791>

### 3.5 Slimmable and universally slimmable networks

Slimmable networks train one model executable at different widths. Universally slimmable networks extend this to arbitrary widths and introduce techniques such as the sandwich rule and in-place distillation.

References:

- Slimmable Neural Networks: <https://arxiv.org/abs/1812.08928>
- Universally Slimmable Networks: <https://arxiv.org/abs/1903.05134>

### 3.6 BERT, RoBERTa, and GLUE

BERT-base and RoBERTa-base are suitable transformer backbones for this project because they are established, fit on one high-end local GPU for fine-tuning, and are widely benchmarked on GLUE-style natural-language-understanding tasks.

References:

- BERT: <https://arxiv.org/abs/1810.04805>
- RoBERTa: <https://arxiv.org/abs/1907.11692>
- GLUE: <https://arxiv.org/abs/1804.07461>

### 3.7 Novelty boundary

The project must not claim:

> We invented multi-depth inference.

The defensible claim is narrower:

> We provide a ground-truth-supervised, statically deployable prefix-training and selection framework, evaluate it using real measured hardware metrics, and show when it does or does not beat strong static-depth baselines.

---

## 4. Target Contributions

A top-tier version of this project should aim for three contributions.

### Contribution 1: Method

A training recipe for **prefix-deployable models** using supervised intermediate heads and depth-sandwich training.

### Contribution 2: Selection protocol

A rigorous post-training selection protocol that chooses the shallowest deployable prefix using held-out validation performance and measured hardware cost.

### Contribution 3: Empirical frontier

A real cost-quality frontier comparing:

1. full model;
2. independently trained shallow models;
3. naïvely truncated deep model;
4. deep supervision with all exits;
5. LayerDrop-style depth robustness;
6. PODS;
7. optional PODS plus width elasticity.

The decisive question is:

> Does PODS produce a better measured quality-cost frontier than training separate shallow models or using known elastic-depth methods?

---

## 5. Hardware Constraint

The assumed local hardware is a single RTX 5090-class GPU. NVIDIA describes the GeForce RTX 5090 as Blackwell-based with 32 GB GDDR7 memory.

Reference:

- NVIDIA RTX 5090 product page: <https://www.nvidia.com/en-gb/geforce/graphics-cards/50-series/rtx-5090/>

This hardware is enough for serious single-GPU experiments, but not for large-scale LLM pretraining.

Appropriate project scope:

1. CIFAR-scale image experiments;
2. BERT-base / RoBERTa-base fine-tuning experiments;
3. optional small language-model diagnostics;
4. no frontier-scale LLM pretraining;
5. no large hyperparameter sweeps unless carefully staged.

---

## 6. Project Name and Paper Framing

Suggested paper title:

> **Prefix Optimised Depth Selection: Training One Model for Static Inference-Cost Selection**

More conservative title:

> **Static Depth Selection from Multi-Exit Supervised Networks**

The conservative title may be better for peer review.

---

## 7. Mathematical Formulation

### 7.1 Model

Let the model consist of \(L\) sequential blocks:

\[
h_0 = E(x)
\]

\[
h_\ell = F_\ell(h_{\ell-1}; \theta_\ell), \qquad \ell = 1, \dots, L
\]

Define candidate deployment-depth set:

\[
\mathcal{D} \subseteq \{1, \dots, L\}
\]

For BERT-base, a natural choice is:

\[
\mathcal{D} = \{2,4,6,8,10,12\}
\]

For ResNet-style models, use stage-aligned depths rather than arbitrary internal blocks.

Each candidate depth has an output head:

\[
\hat{y}_d = H_d(h_d; \phi_d)
\]

The prefix model at depth \(d\) is:

\[
M_d(x) = H_d(F_d \circ \cdots \circ F_1(E(x)))
\]

At deployment, only one \(d\) is selected.

---

### 7.2 Supervised task loss

For classification:

\[
\ell_d(x,y) = \operatorname{CE}(H_d(h_d), y)
\]

For regression:

\[
\ell_d(x,y) = \|H_d(h_d)-y\|_2^2
\]

For STS-style semantic similarity:

\[
\ell_d(x,y) = \operatorname{MSE}(H_d(h_d), y)
\]

For token classification:

\[
\ell_d(x,y) = \frac{1}{T} \sum_{t=1}^T \operatorname{CE}(H_d(h_{d,t}), y_t)
\]

The primary project should use ground-truth task loss, not final-model agreement.

---

### 7.3 Basic multi-depth objective

The simplest training objective is:

\[
\mathcal{J}_{all}
=
\mathbb{E}_{(x,y)\sim\mathcal{S}_{train}}
\left[
\sum_{d \in \mathcal{D}} \alpha_d \ell_d(x,y)
\right]
\]

where \(\alpha_d\) controls how much each prefix matters.

A naïve uniform choice is:

\[
\alpha_d = 1
\]

but this may over-constrain early layers.

A safer depth-weighted choice is:

\[
\alpha_d = \left(\frac{d}{L}\right)^p
\]

with:

\[
p \in \{0,1,2\}
\]

tested as an ablation.

---

### 7.4 Prefix-sampling objective

Instead of computing every exit on every batch, sample deployable depths:

\[
d \sim q(d)
\]

and train:

\[
\mathcal{J}_{sampled}
=
\mathbb{E}_{(x,y)}
\mathbb{E}_{d\sim q(d)}
[
\alpha_d \ell_d(x,y)
]
\]

The stronger practical variant is **depth sandwich training**:

\[
\mathcal{B}_{depth}
=
\{d_{min}, d_{max}\}
\cup
\operatorname{SampleK}(\mathcal{D}\setminus\{d_{min},d_{max}\})
\]

Each batch trains:

1. the shallowest candidate;
2. the full model;
3. one or more random middle depths.

This is inspired by the sandwich rule from universally slimmable networks, but applied to static depth-prefix training rather than width elasticity.

---

### 7.5 Static deployment selection

After training, evaluate each prefix on validation data.

For a metric to be maximised, such as accuracy or F1:

\[
d^*
=
\arg\min_{d\in\mathcal{D}}
C_{measured}(d)
\]

subject to:

\[
Q_{val}(d) \ge Q_{target}
\]

For loss:

\[
d^*
=
\arg\min_{d\in\mathcal{D}}
C_{measured}(d)
\]

subject to:

\[
L_{val}(d) \le L_{target}
\]

The target can be absolute:

\[
Q_{target}=0.90
\]

or relative to the full model:

\[
Q_{target}=Q_{val}(L)-\epsilon
\]

with \(\epsilon\) chosen before test evaluation.

For the paper, use several target bands:

\[
\epsilon \in \{0.0, 0.0025, 0.005, 0.01, 0.02\}
\]

For each target, select the shallowest measured-cost prefix.

---

## 8. Critical Correction: Do Not Select Using Training Accuracy

Engineers should not choose the deployed depth using training accuracy.

Use:

1. training set for optimisation;
2. validation set for depth selection and hyperparameter choice;
3. final test set for one-time reporting.

Training accuracy is allowed only as a diagnostic.

For GLUE-style tasks, because official test labels are generally not available during local experimentation, use a nested split:

1. split the original training set into train and selection-validation;
2. use the official development set as the final held-out evaluation;
3. never tune after inspecting final held-out results.

---

## 9. Experimental Scope

The project should have three levels.

### Level A: Fast sanity experiments

Purpose: prove the machinery works.

Datasets:

1. MNIST or sklearn digits;
2. CIFAR-10;
3. CIFAR-100.

Models:

1. MLP for debugging;
2. small CNN;
3. ResNet-style model.

These are not enough for the final paper, but they are essential for catching implementation bugs.

---

### Level B: Serious image experiments

Purpose: test static prefix deployment in a controlled non-language setting.

Datasets:

1. CIFAR-10;
2. CIFAR-100.

Models:

1. ResNet-56;
2. ResNet-110, if feasible;
3. small ViT if feasible;
4. ConvNeXt-Tiny only if the single-GPU budget allows.

Candidate depths should be stage-aligned. For ResNet, avoid cutting in the middle of a residual stage unless the implementation is clean and the head is well-defined.

---

### Level C: Serious NLP experiments

Purpose: test the idea where transformer depth matters.

Datasets:

Use GLUE tasks that fit single-GPU fine-tuning:

1. SST-2;
2. MRPC;
3. QQP;
4. QNLI;
5. RTE;
6. CoLA;
7. STS-B;
8. MNLI, if runtime allows.

Models:

1. BERT-base;
2. RoBERTa-base;
3. optionally DistilBERT as a shallow-model baseline.

Candidate depths:

\[
\mathcal{D} = \{2,4,6,8,10,12\}
\]

Heads:

1. classification: CLS pooling + layer norm + linear classifier;
2. regression: CLS pooling + layer norm + scalar head;
3. optionally mean pooling as an ablation.

---

## 10. Main Baselines

The paper needs strong baselines. Weak baselines will not hold up.

### Baseline 1: Full model

Train and deploy full depth:

\[
M_L
\]

This is the quality anchor.

---

### Baseline 2: Independently trained shallow models

For each candidate depth \(d\), train a separate model of exactly depth \(d\).

This is the hardest baseline.

If PODS at depth \(d\) is worse than a separately trained depth-\(d\) model, the remaining argument must be:

> PODS gives a whole deployment frontier from one training run.

That is useful, but weaker than claiming that PODS dominates shallow specialists.

---

### Baseline 3: Naïve truncation

Train only the full-depth model. Attach or train a shallow head after training and evaluate depth \(d\).

Variants:

1. frozen trunk + trained head;
2. fine-tuned shallow head;
3. direct untrained intermediate head, only as a diagnostic.

This tests whether multi-depth training is actually necessary.

---

### Baseline 4: All-exit deep supervision

Train:

\[
\sum_{d\in\mathcal{D}} \ell_d
\]

with uniform weights.

This distinguishes PODS's training schedule from the obvious simple version.

---

### Baseline 5: LayerDrop-style training

LayerDrop is a direct transformer-depth baseline because it trains models so subnetworks of different depths can be selected at inference time.

Implement either:

1. original-style structured layer dropout;
2. depth-prefix dropout only;
3. available fairseq / Hugging Face equivalent if cleanly reproducible.

---

### Baseline 6: Once-for-All-style progressive shrinking

OFA is a direct elastic-network baseline because it decouples training from subnetwork selection and supports different depth/width settings.

For this project, implement a simplified depth-only progressive shrinking baseline.

---

### Baseline 7: Distillation baseline

Train a shallow depth-\(d\) model using:

1. ground-truth labels;
2. optional soft targets from the full model.

This is not the main philosophical objective, but reviewers will expect it.

---

## 11. Method Variants to Test

### PODS-A: all-depth supervised training

\[
\mathcal{J}_{all}
=
\sum_{d\in\mathcal{D}}\alpha_d\ell_d
\]

---

### PODS-B: depth sandwich training

Every batch trains:

1. shallowest depth;
2. deepest depth;
3. one or two random intermediate depths.

---

### PODS-C: progressive depth training

Training is staged:

1. train full depth only;
2. add deeper prefixes;
3. add shallower prefixes;
4. fine-tune with depth sandwich.

Hypothesis: early full-depth warmup prevents shallow exits from damaging representation learning.

---

### PODS-D: shared head versus independent heads

Independent heads:

\[
H_d \neq H_{d'}
\]

Shared head:

\[
H_d = H
\]

For transformers, the shared-head variant may work only if hidden states at different depths are similarly normalised.

---

### PODS-E: head architecture

Test:

1. linear head;
2. layer norm + linear;
3. small MLP head;
4. pooled hidden state choices for transformers.

The head must remain small. If the head becomes too powerful, the prefix comparison becomes unfair.

---

## 12. Real Metrics Only

### 12.1 Quality metrics

Use the standard measured metric for each task.

For classification:

1. accuracy;
2. macro-F1 where class imbalance matters;
3. Matthews correlation for CoLA;
4. task-specific GLUE metric where applicable.

For regression:

1. mean squared error;
2. Pearson/Spearman for STS-B.

For loss:

1. validation cross-entropy;
2. validation negative log likelihood;
3. validation MSE.

All quality values must be computed on real examples.

---

### 12.2 Deployment metrics

For every selected prefix and baseline, measure:

1. p50 latency, batch size 1;
2. p95 latency, batch size 1;
3. throughput, examples/sec at realistic batch sizes;
4. peak GPU memory allocated;
5. model file size on disk;
6. optional GPU-reported energy.

Do not use theoretical FLOPs as a primary result.

Derived ratios are allowed only when derived from real measurements, for example:

\[
\text{latency reduction}
=
\frac{T_{full}-T_{prefix}}{T_{full}}
\]

where both \(T\) values are measured.

---

### 12.3 Measurement protocol

For latency:

1. use fixed input shapes;
2. warm up the GPU;
3. use `torch.cuda.synchronize()`;
4. use CUDA events or a carefully synchronised wall-clock timer;
5. exclude dataloader/tokenizer time for model-only latency;
6. separately report end-to-end latency including preprocessing if desired;
7. repeat enough times to estimate p50/p95;
8. keep GPU clocks/power settings stable if possible;
9. report CUDA, PyTorch, driver, OS, and GPU model.

For NLP, benchmark at:

1. sequence length 64;
2. sequence length 128;
3. sequence length 256;
4. optional sequence length 512.

For batch sizes:

1. 1;
2. 8;
3. 32;
4. largest batch that fits memory without OOM.

---

## 13. Repository Structure

```text
pods/
  configs/
    cifar/
    glue/
    benchmark/
  data/
    README.md
  src/
    models/
      mlp.py
      resnet_pods.py
      bert_pods.py
      roberta_pods.py
      heads.py
      prefix_extract.py
    training/
      train_full.py
      train_pods_all.py
      train_pods_sandwich.py
      train_layerdrop.py
      train_shallow.py
      train_distill.py
    eval/
      eval_quality.py
      eval_prefixes.py
      select_prefix.py
      benchmark_latency.py
      benchmark_memory.py
      bootstrap.py
    utils/
      seeds.py
      logging.py
      metrics.py
      cuda_timing.py
      split_data.py
  scripts/
    run_sanity.sh
    run_cifar.sh
    run_glue.sh
    run_benchmarks.sh
    make_tables.sh
  paper/
    main.tex
    sections/
    figures/
    tables/
  results/
    raw/
    processed/
  README.md
  environment.yml
  pyproject.toml
```

All reported tables should be generated from raw logs, not manually edited.

---

## 14. Step-by-Step Project Plan with Success Criteria

### Step 0: Freeze the research question

Write a one-page internal memo containing:

1. exact claim;
2. exact non-claim;
3. baseline list;
4. datasets;
5. metrics;
6. selection protocol.

Success criteria:

1. the claim does not say “dynamic early exit”;
2. the claim does not use training accuracy for deployment selection;
3. the claim names LayerDrop, OFA, and independently trained shallow models as baselines;
4. the memo has a negative-result fallback.

Negative-result fallback:

> If PODS does not beat strong baselines, the paper becomes a rigorous empirical study of when static prefix deployment fails.

---

### Step 1: Literature matrix

Create a table with these rows:

1. BranchyNet;
2. DeeBERT;
3. LayerDrop;
4. Once-for-All;
5. Slimmable networks;
6. Universally slimmable networks;
7. MatFormer;
8. distillation;
9. structured pruning;
10. early-exit transformers;
11. layer dropping in LLMs;
12. model compression surveys;
13. dynamic routing methods;
14. adaptive computation time;
15. static subnetwork selection.

Columns:

1. dynamic or static;
2. depth, width, or both;
3. supervised labels or distillation;
4. per-example routing or fixed deployment;
5. hardware-measured metrics;
6. whether shallow baselines are trained separately;
7. what gap remains.

Success criteria:

1. at least 15 relevant papers included;
2. every baseline in the experiment maps to prior work;
3. novelty statement is reduced to something defensible.

---

### Step 2: Data protocol

For CIFAR:

1. train: original train minus validation;
2. selection-validation: held-out subset of train;
3. final test: official test set.

For GLUE-style tasks:

1. train: split from official training data;
2. selection-validation: held-out from official training data;
3. final held-out: official development set.

Success criteria:

1. split files saved with hashes;
2. no test/final-held-out metric used during hyperparameter tuning;
3. all seeds recorded;
4. class distributions logged.

Gotcha:

Small GLUE tasks such as RTE and MRPC have high variance. They need multiple seeds and confidence intervals.

---

### Step 3: Implement prefix model abstraction

Every model must expose:

```python
model.forward_prefix(x, depth=d)
model.forward_all_prefixes(x, depths=D)
model.extract_prefix(depth=d)
model.available_depths
```

For transformers:

```python
hidden_states = encoder(..., output_hidden_states=True)
logits_d = head_d(hidden_states[d])
```

However, this hidden-state method is only for analysis or training convenience. For deployment benchmarking, it is not acceptable because it usually computes all layers.

For deployment extraction:

```python
prefix = PrefixModel(
    embedding=model.embedding,
    blocks=model.blocks[:d],
    head=model.heads[d],
)
```

Success criteria:

1. extracted prefix produces numerically identical logits to the full PODS model evaluated at the same depth;
2. unit test confirms no layer \(>d\) is executed;
3. exported prefix has smaller model file size than full model;
4. benchmark script can run extracted prefix independently.

Gotcha:

For Hugging Face models, simply using `output_hidden_states=True` still runs all layers. For deployment benchmarking, instantiate a shorter model or modify the encoder loop to stop at \(d\).

---

### Step 4: Implement measurement harness

Latency pseudocode:

```python
def measure_latency(model, batch, warmup=100, repeats=1000):
    model.eval()
    latencies_ms = []

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(batch)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for _ in range(repeats):
            start_event.record()
            _ = model(batch)
            end_event.record()
            torch.cuda.synchronize()
            latencies_ms.append(start_event.elapsed_time(end_event))

    return {
        "p50_ms": np.percentile(latencies_ms, 50),
        "p95_ms": np.percentile(latencies_ms, 95),
        "mean_ms": np.mean(latencies_ms),
        "std_ms": np.std(latencies_ms),
    }
```

Throughput pseudocode:

```python
def measure_throughput(model, dataloader, batches=200):
    model.eval()
    n_examples = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i == batches:
                break
            batch = move_to_cuda(batch)
            _ = model(batch)
            n_examples += batch_size(batch)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return n_examples / (t1 - t0)
```

Memory pseudocode:

```python
def measure_peak_memory(model, batch):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model.eval()
    with torch.no_grad():
        _ = model(batch)
        torch.cuda.synchronize()

    return torch.cuda.max_memory_allocated()
```

Success criteria:

1. latency coefficient of variation is acceptably stable after warmup;
2. benchmark uses extracted prefix, not full model hidden-state extraction;
3. p50 and p95 are both reported;
4. throughput is measured on real batches;
5. model-only and end-to-end settings are clearly separated.

---

### Step 5: Sanity MLP experiment

Purpose:

Validate the entire training/evaluation/selection pipeline.

Architecture:

1. depth \(L=8\);
2. hidden dimension 256;
3. heads at depths \(\{1,2,4,6,8\}\).

Training variants:

1. full only;
2. all-depth;
3. depth sandwich.

Success criteria:

1. all variants train without NaNs;
2. deeper prefixes generally improve validation quality;
3. selected prefix can be extracted and benchmarked;
4. at least one shallow prefix has lower measured latency than full model;
5. all result tables are generated automatically.

This stage is not publishable; it is a machinery test.

---

### Step 6: CIFAR image experiment

Models:

1. ResNet-56;
2. ResNet-110, if feasible.

Candidate depths:

Use stage-aligned prefixes. Example:

\[
\mathcal{D} = \{1/3, 2/3, 1.0\}
\]

or block counts:

\[
\mathcal{D} = \{6, 12, 18, 24, 30\}
\]

depending on the exact architecture.

Training variants:

1. full model;
2. independently trained shallow models;
3. naïve truncation;
4. all-depth supervision;
5. LayerDrop-style;
6. PODS all-depth weighted;
7. PODS depth sandwich;
8. optional PODS progressive.

Success criteria:

1. full model reaches a credible CIFAR baseline;
2. PODS prefixes beat naïve truncation at matched depth;
3. PODS frontier is competitive with independently trained shallow models;
4. measured latency and memory are reported for every prefix;
5. at least 3 random seeds per main configuration.

Gotcha:

BatchNorm can break prefix extraction. Either:

1. use GroupNorm/LayerNorm;
2. maintain separate BatchNorm statistics per candidate depth;
3. recalibrate BatchNorm statistics for each extracted prefix using training data without gradient updates.

If using BatchNorm, BN handling must be an ablation.

---

### Step 7: BERT/RoBERTa PODS implementation

Models:

1. BERT-base;
2. RoBERTa-base.

Candidate depths:

\[
\mathcal{D} = \{2,4,6,8,10,12\}
\]

Head:

```python
class PrefixClassificationHead(nn.Module):
    def __init__(self, hidden_dim, num_labels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(hidden_dim, num_labels)

    def forward(self, hidden):
        cls = hidden[:, 0]
        return self.out(self.dropout(self.norm(cls)))
```

Training modes:

1. full-depth fine-tuning;
2. independent shallow fine-tuning;
3. all-depth supervised;
4. PODS depth sandwich;
5. LayerDrop-style baseline.

Success criteria:

1. extracted BERT prefix matches in-model depth-\(d\) logits;
2. full-depth PODS does not collapse relative to full-depth fine-tuning;
3. at least three GLUE tasks run end-to-end before expanding;
4. final version covers at least six GLUE tasks;
5. final results use at least three seeds for main comparisons.

Gotcha:

Do not compare a PODS depth-6 prefix against full BERT-base only. Compare it against a separately fine-tuned 6-layer BERT-style model.

---

### Step 8: Depth sandwich training algorithm

Pseudocode:

```python
def train_pods_sandwich(model, loader, depths, optimizer, cfg):
    d_min = min(depths)
    d_max = max(depths)

    for batch in loader:
        x, y = batch
        x, y = move_to_device(x, y)

        sampled = random.sample(
            [d for d in depths if d not in [d_min, d_max]],
            k=cfg.num_middle_depths,
        )

        batch_depths = [d_min] + sampled + [d_max]

        total_loss = 0.0
        logs = {}

        for d in batch_depths:
            logits = model.forward_prefix(x, depth=d)
            loss_d = task_loss(logits, y)

            alpha_d = depth_weight(d, d_max, cfg.weight_power)
            total_loss = total_loss + alpha_d * loss_d

            logs[f"loss_depth_{d}"] = loss_d.item()

        total_loss = total_loss / len(batch_depths)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        log(logs)
```

Depth weight:

```python
def depth_weight(d, d_max, p):
    return (d / d_max) ** p
```

Ablate:

```text
p ∈ {0, 1, 2}
num_middle_depths ∈ {0, 1, 2, all}
```

Success criteria:

1. no candidate depth is starved of updates;
2. training logs show every depth's validation metric;
3. full-depth performance remains close to the full-depth baseline;
4. shallow-depth performance improves over naïve truncation.

---

### Step 9: Prefix evaluation

Pseudocode:

```python
@torch.no_grad()
def evaluate_all_prefixes(model, dataloader, depths, metric_fn):
    results = {}

    for d in depths:
        prefix = model.extract_prefix(depth=d).eval().cuda()

        losses = []
        preds = []
        labels = []

        for batch in dataloader:
            x, y = move_to_device(batch)
            logits = prefix(x)
            loss = task_loss(logits, y, reduction="none")

            losses.extend(loss.detach().cpu().tolist())
            preds.extend(prediction_from_logits(logits))
            labels.extend(y.detach().cpu().tolist())

        results[d] = {
            "loss": float(np.mean(losses)),
            **metric_fn(preds, labels),
        }

        del prefix
        torch.cuda.empty_cache()

    return results
```

Success criteria:

1. every prefix is evaluated using the extracted model;
2. metrics are computed on the same held-out data;
3. no prefix uses hidden states from deeper layers;
4. output file includes raw predictions for reproducibility.

---

### Step 10: Static prefix selection

Pseudocode:

```python
def select_prefix(results, target, cost_key, quality_key):
    feasible = []

    for d, r in results.items():
        if r[quality_key] >= target:
            feasible.append((d, r[cost_key], r[quality_key]))

    if not feasible:
        return max(results.keys())

    feasible.sort(key=lambda x: x[1])
    return feasible[0][0]
```

More robust version with bootstrap confidence intervals:

```python
def select_prefix_ci(results, target, cost_key, quality_lcb_key):
    feasible = []

    for d, r in results.items():
        if r[quality_lcb_key] >= target:
            feasible.append((d, r[cost_key]))

    if not feasible:
        return max(results.keys())

    return min(feasible, key=lambda x: x[1])[0]
```

Success criteria:

1. selected depth is chosen before final test evaluation;
2. selection rule is fixed in config;
3. all selected prefixes are then evaluated once on final test;
4. paper reports both validation-selected depth and final test performance.

---

### Step 11: Bootstrap and uncertainty

For quality metrics:

1. bootstrap examples from the final evaluation set;
2. compute 95% confidence intervals;
3. report mean and interval.

For latency:

1. repeat benchmark runs;
2. report p50, p95, and confidence interval over repeated runs;
3. do not mix quality bootstrap with latency bootstrap.

Pseudocode:

```python
def bootstrap_metric(y_true, y_pred, metric_fn, n_boot=10000):
    n = len(y_true)
    values = []

    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        values.append(metric_fn(y_true[idx], y_pred[idx]))

    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }
```

Success criteria:

1. all main tables include uncertainty;
2. pairwise comparisons use paired seeds or paired examples where possible;
3. no claim of superiority is made from a single seed unless labelled exploratory.

---

### Step 12: Main ablations

Minimum ablation grid:

```text
Training objective:
  full-only
  all-depth uniform
  all-depth weighted
  depth sandwich
  progressive depth sandwich

Candidate depth set:
  every 2 layers
  every 3 layers
  only shallow/mid/full

Head:
  linear
  layer norm + linear
  MLP

Depth weight:
  p = 0
  p = 1
  p = 2

Sampling:
  min + max only
  min + max + 1 middle
  min + max + 2 middle
  all depths

Selection criterion:
  absolute target
  relative-to-full target
  confidence-bound target

Baseline:
  independent shallow
  naïve truncation
  LayerDrop
  simplified OFA/progressive shrinking
```

Success criteria:

1. every ablation answers a named question;
2. no ablation is included only because it looks good;
3. failed ablations are reported in appendix;
4. final method is not chosen using final test performance.

---

## 15. Gotchas and Failure Modes

### Gotcha 1: the idea may lose to shallow models

If an independently trained 6-layer model beats the 6-layer PODS prefix, the claim must shift.

Possible valid claim:

> PODS gives a useful multi-depth deployment frontier from one training run, but independently trained shallow specialists remain stronger at individual depths.

Invalid claim:

> PODS is better than shallow models.

---

### Gotcha 2: auxiliary heads can damage full-depth performance

Early heads force early layers to become predictive. This may conflict with deep representation learning.

Mitigations:

1. weight deeper exits more;
2. warm up full-depth first;
3. use depth sandwich instead of all exits every batch;
4. add shallow heads only after initial fine-tuning;
5. use smaller heads.

---

### Gotcha 3: naïve hidden-state extraction is not deployment

For BERT, this is wrong:

```python
outputs = bert(x, output_hidden_states=True)
logits = head_6(outputs.hidden_states[6])
```

That still computed all 12 layers.

Correct deployment benchmark:

```python
prefix = extract_bert_prefix(model, depth=6)
logits = prefix(x)
```

---

### Gotcha 4: BatchNorm contamination

If image models use BatchNorm, one set of BN statistics may not fit all depths.

Mitigations:

1. use GroupNorm;
2. use separate BN per depth;
3. recalibrate BN after extracting each prefix;
4. report which one is used.

---

### Gotcha 5: validation leakage

If \(d\), hyperparameters, and stopping time are chosen on the same validation set repeatedly, the final result is biased.

Mitigation:

1. split training into train and selection-validation;
2. reserve final test;
3. use final test once.

---

### Gotcha 6: measured latency may not improve as expected

Small models can be memory-bound or overhead-bound.

Mitigations:

1. benchmark real extracted prefixes;
2. test multiple batch sizes;
3. use `torch.compile` as a separate benchmark condition;
4. optionally export to ONNX/TensorRT;
5. avoid claiming cost reduction from parameter count alone.

---

### Gotcha 7: full-depth anchor may dominate training

If \(d=L\) gets too much weight, shallow prefixes underperform.

Mitigation:

1. tune \(\alpha_d\);
2. use equal sampling but weighted losses;
3. report prefix-specific training loss curves.

---

### Gotcha 8: shallow prefixes may overfit small tasks

Especially on RTE and MRPC.

Mitigation:

1. multiple seeds;
2. dropout;
3. weight decay;
4. early stopping;
5. confidence intervals.

---

### Gotcha 9: GLUE task metrics are heterogeneous

Do not average raw metrics blindly unless the averaging rule is defined before experiments.

Mitigation:

1. report each task separately;
2. optionally report macro-average of normalised task scores;
3. keep task-level tables primary.

---

### Gotcha 10: one GPU limits the claim

A single RTX 5090 can support a serious paper, but the paper should avoid claims about frontier-scale LLMs unless tested. The claim should be about static deployable prefixes in supervised fine-tuning and moderate-scale architectures, not universal LLM scaling.

---

## 16. Optimisation Details for the RTX 5090 Setup

Use:

1. BF16 or FP16 mixed precision;
2. gradient accumulation for larger effective batch sizes;
3. cached tokenisation for GLUE;
4. fixed max sequence length for benchmarking;
5. dynamic padding for training only;
6. PyTorch scaled-dot-product attention where available;
7. `torch.compile` as a benchmark condition, not necessarily the default;
8. pinned-memory dataloaders;
9. `set_to_none=True` in optimiser zeroing;
10. gradient clipping for transformer fine-tuning.

Avoid:

1. computing all hidden states during deployment benchmarks;
2. using dataloader overhead in model-only latency;
3. using different preprocessing across methods;
4. selecting hyperparameters from final test results.

---

## 17. Paper-Grade Experiment Matrix

### 17.1 CIFAR table

For each model and method:

```text
Method
Depth
Validation accuracy
Test accuracy
Test loss
p50 latency, batch 1
p95 latency, batch 1
Throughput, batch 32
Peak GPU memory
Model file size
Seeds
```

---

### 17.2 GLUE table

For each task and method:

```text
Task
Model
Method
Selected depth
Validation-selection metric
Final held-out metric
Final held-out loss
p50 latency, seq len 128, batch 1
p95 latency, seq len 128, batch 1
Throughput, seq len 128, batch 32
Peak GPU memory
Model file size
Seeds
```

---

### 17.3 Frontier plot

For each task:

\[
x = \text{measured p50 latency}
\]

\[
y = \text{final held-out task metric}
\]

Curves:

1. independent shallow;
2. naïve truncation;
3. LayerDrop;
4. PODS;
5. full model.

A strong result is not one cherry-picked point. It is a better frontier.

---

## 18. Success Criteria for the Full Project

The project is scientifically successful if it reaches one of two outcomes.

### Positive-result success

The paper can claim PODS is useful if:

1. PODS beats naïve truncation on most evaluated task/model pairs;
2. PODS is competitive with independently trained shallow models at matched measured latency;
3. PODS provides a usable multi-depth frontier from a single training run;
4. selected prefixes show real measured latency or throughput improvement;
5. full-depth performance is not materially degraded;
6. results hold across multiple seeds;
7. baselines are strong and fairly tuned.

---

### Negative-result success

The paper can still be valuable if it shows:

1. static prefix deployment often fails against independently trained shallow models;
2. auxiliary losses harm full-depth performance;
3. theoretical layer reduction does not reliably translate to measured latency;
4. LayerDrop/OFA-style baselines dominate simple supervised prefix training;
5. the failure modes are carefully measured and reproducible.

A rigorous negative result can be publishable. A weak positive result cannot.

---

## 19. What Would Make This NeurIPS/ICML/JMLR-Worthy?

The current idea alone is probably not enough, because LayerDrop, OFA, slimmable networks, and MatFormer already cover nearby ground. The project becomes stronger if it delivers one of these:

### Strong angle A: static deployment beats dynamic early exit in production realism

Show that fixed selected prefixes give simpler batching, lower memory, and real latency wins compared with dynamic methods.

### Strong angle B: ground-truth-supervised prefix training beats distillation-based elastic methods

This would be interesting because the objective does not assume the full model is the teacher.

### Strong angle C: prefix training produces better depth frontiers than LayerDrop

This is the cleanest transformer claim.

### Strong angle D: careful measurement invalidates common FLOP-based claims

This could be a strong empirical paper if the hardware measurements are rigorous.

### Strong angle E: one training run gives many deployable prefixes with little loss versus separately trained specialists

This is closest to the OFA value proposition, but focused on supervised static depth and transformer fine-tuning.

---

## 20. Full Method Pseudocode

### 20.1 Model wrapper

```python
class PODSTransformer(nn.Module):
    def __init__(self, base_model, candidate_depths, num_labels, task_type):
        super().__init__()
        self.embeddings = base_model.embeddings
        self.layers = nn.ModuleList(base_model.encoder.layer)
        self.candidate_depths = candidate_depths
        self.task_type = task_type

        hidden = base_model.config.hidden_size

        self.heads = nn.ModuleDict({
            str(d): make_head(hidden, num_labels, task_type)
            for d in candidate_depths
        })

    def forward_prefix(self, input_ids, attention_mask, depth):
        h = self.embeddings(input_ids=input_ids)

        extended_mask = make_extended_attention_mask(attention_mask)

        for i in range(depth):
            h = self.layers[i](h, extended_mask)[0]

        return self.heads[str(depth)](h)

    def forward_depths(self, input_ids, attention_mask, depths):
        h = self.embeddings(input_ids=input_ids)
        extended_mask = make_extended_attention_mask(attention_mask)

        logits = {}

        for i, layer in enumerate(self.layers, start=1):
            h = layer(h, extended_mask)[0]

            if i in depths:
                logits[i] = self.heads[str(i)](h)

        return logits

    def extract_prefix(self, depth):
        return ExtractedPrefixTransformer(
            embeddings=copy.deepcopy(self.embeddings),
            layers=copy.deepcopy(self.layers[:depth]),
            head=copy.deepcopy(self.heads[str(depth)]),
            depth=depth,
            task_type=self.task_type,
        )
```

---

### 20.2 Training loop

```python
def train_pods(
    model,
    train_loader,
    val_loader,
    candidate_depths,
    optimizer,
    scheduler,
    cfg,
):
    best_state = None
    best_val_score = -float("inf")

    for epoch in range(cfg.epochs):
        model.train()

        for batch in train_loader:
            batch = move_to_cuda(batch)

            if cfg.mode == "all":
                depths = candidate_depths

            elif cfg.mode == "sandwich":
                depths = sample_depth_sandwich(
                    candidate_depths,
                    num_middle=cfg.num_middle_depths,
                )

            elif cfg.mode == "full_only":
                depths = [max(candidate_depths)]

            else:
                raise ValueError(cfg.mode)

            logits_by_depth = model.forward_depths(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                depths=depths,
            )

            total_loss = 0.0

            for d, logits in logits_by_depth.items():
                loss_d = task_loss(
                    logits,
                    batch["labels"],
                    task_type=cfg.task_type,
                )

                alpha_d = (d / max(candidate_depths)) ** cfg.weight_power
                total_loss = total_loss + alpha_d * loss_d

            total_loss = total_loss / len(depths)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.grad_clip,
                )

            optimizer.step()
            scheduler.step()

        val_results = evaluate_all_prefixes(
            model,
            val_loader,
            candidate_depths,
            metric_fn=cfg.metric_fn,
        )

        selection_score = val_results[max(candidate_depths)][cfg.primary_metric]

        if selection_score > best_val_score:
            best_val_score = selection_score
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model
```

---

### 20.3 Selection and final evaluation

```python
def run_selection_and_final_eval(
    model,
    candidate_depths,
    selection_loader,
    final_loader,
    benchmark_batches,
    cfg,
):
    selection_quality = evaluate_all_prefixes(
        model,
        selection_loader,
        candidate_depths,
        cfg.metric_fn,
    )

    measured_costs = {}

    for d in candidate_depths:
        prefix = model.extract_prefix(d).cuda().eval()

        measured_costs[d] = {
            **measure_latency(prefix, benchmark_batches[d]["latency"]),
            "throughput": measure_throughput(
                prefix,
                benchmark_batches[d]["throughput"],
            ),
            "peak_memory": measure_peak_memory(
                prefix,
                benchmark_batches[d]["memory"],
            ),
            "model_bytes": serialized_model_size(prefix),
        }

    full_depth = max(candidate_depths)
    full_score = selection_quality[full_depth][cfg.primary_metric]
    target = full_score - cfg.allowed_metric_drop

    selected_depth = select_prefix(
        merge(selection_quality, measured_costs),
        target=target,
        cost_key=cfg.selection_cost_key,
        quality_key=cfg.primary_metric,
    )

    selected_prefix = model.extract_prefix(selected_depth).cuda().eval()

    final_quality = evaluate_model(
        selected_prefix,
        final_loader,
        cfg.metric_fn,
    )

    final_cost = benchmark_prefix(
        selected_prefix,
        benchmark_batches[selected_depth],
    )

    return {
        "selected_depth": selected_depth,
        "selection_quality": selection_quality,
        "measured_costs": measured_costs,
        "final_quality": final_quality,
        "final_cost": final_cost,
    }
```

---

## 21. Paper Outline

### Abstract

Must include:

1. static deployment-depth selection;
2. ground-truth supervised multi-depth training;
3. real hardware measurements;
4. main empirical finding;
5. limitations.

Do not imply dynamic routing unless explicitly contrasting against it.

---

### Section 1: Introduction

Explain:

1. deployment constraints require fixed, efficient models;
2. training separate models for each budget is expensive;
3. dynamic early exit complicates batching and memory;
4. static prefix deployment gives fixed graphs and real model-size reduction;
5. question: can one training run produce many good prefixes?

---

### Section 2: Related work

Subsections:

1. early exit and branch networks;
2. adaptive depth in transformers;
3. LayerDrop and structured dropout;
4. once-for-all and slimmable networks;
5. transformer elastic inference and MatFormer;
6. model compression and distillation.

---

### Section 3: Method

Include:

1. prefix model definition;
2. candidate depth set;
3. supervised prefix heads;
4. all-depth objective;
5. depth sandwich objective;
6. static selection algorithm;
7. extraction and deployment.

---

### Section 4: Experimental protocol

Include:

1. datasets and splits;
2. models;
3. baselines;
4. metrics;
5. real measurement protocol;
6. seed protocol;
7. hyperparameter tuning rules;
8. hardware/software environment.

---

### Section 5: Results

Core figures:

1. quality versus measured latency frontier;
2. quality versus model size;
3. selected-depth table;
4. comparison against shallow models;
5. comparison against LayerDrop/OFA-style baselines;
6. seed variability.

---

### Section 6: Ablations

Include:

1. loss weighting;
2. head type;
3. depth sampling;
4. candidate depth set;
5. BN handling for image models;
6. validation-selection rule;
7. `torch.compile` or deployment-backend condition.

---

### Section 7: Discussion

Discuss:

1. when PODS works;
2. when it fails;
3. why shallow specialists may beat prefixes;
4. training cost versus deployment flexibility;
5. static selection versus dynamic early exit;
6. implications for larger LLMs.

---

### Section 8: Limitations

Must include:

1. single-GPU scale;
2. no frontier LLM pretraining;
3. dataset limitations;
4. hardware-specific latency;
5. possible under-tuned baselines;
6. environmental and compute cost disclosure.

---

### Section 9: Reproducibility statement

Include:

1. code release;
2. exact configs;
3. seeds;
4. data splits;
5. raw logs;
6. hardware details;
7. benchmark scripts;
8. model checkpoints where feasible.

---

## 22. Final Acceptance Checklist

### Claim: PODS reduces inference cost

Required evidence:

1. extracted prefix benchmarked;
2. p50/p95 latency measured;
3. throughput measured;
4. peak memory measured;
5. model file size measured;
6. comparison against full model and shallow baseline.

---

### Claim: PODS preserves quality

Required evidence:

1. final held-out metric;
2. final held-out loss;
3. multiple seeds;
4. confidence intervals;
5. task-level table;
6. no selection on final test.

---

### Claim: PODS is better than naïve truncation

Required evidence:

1. same architecture;
2. same candidate depths;
3. same selection protocol;
4. same measurement harness;
5. paired statistical comparison.

---

### Claim: PODS is competitive with training shallow models

Required evidence:

1. independently trained shallow model at each candidate depth;
2. same data;
3. same hyperparameter budget where possible;
4. measured latency and quality;
5. explicit comparison of training cost versus deployment frontier.

---

### Claim: PODS is useful for transformers

Required evidence:

1. BERT-base or RoBERTa-base experiments;
2. multiple GLUE tasks;
3. LayerDrop baseline;
4. extracted-prefix benchmarking;
5. no hidden-state shortcut.

---

## 23. Hardest Reviewer Objections

### Objection 1: This is just early exit

Response:

No. The deployed model is a statically selected prefix with fixed graph, reduced parameter memory, simple batching, and no per-example router.

---

### Objection 2: This is just LayerDrop

Response:

LayerDrop trains layer robustness through structured dropout. PODS trains explicitly supervised deployable prefixes with per-depth heads and selects using measured hardware cost. The paper must empirically compare them.

---

### Objection 3: Why not train a shallow model?

Response:

That is the main baseline. PODS must either match shallow specialists or justify itself by producing a full deployment frontier from one training run.

---

### Objection 4: Your speedup is theoretical

Response:

The paper reports measured p50/p95 latency, throughput, peak memory, and model file size using extracted prefixes.

---

### Objection 5: You tuned on the test set

Response:

Use locked train, selection-validation, and final-held-out splits. Final test is evaluated once.

---

## 24. Most Likely Outcome

The most likely scientifically honest result is mixed:

1. PODS will probably beat naïve truncation.
2. PODS may or may not beat LayerDrop.
3. Independently trained shallow specialists may be stronger at individual depths.
4. PODS's strongest value may be producing a deployment frontier from one training run.
5. Real latency gains will depend heavily on the model, batch size, and backend.

That is still a useful project if measured carefully.

The paper becomes strong if PODS's frontier is consistently close to separately trained shallow models while requiring substantially less total training and producing multiple deployable depths from one model.

---

## 25. Minimal Milestone Schedule

This is not a promise of calendar duration. It is the logical order of execution.

### Milestone 1: Machinery

Deliverables:

1. repository skeleton;
2. seed control;
3. split generation;
4. prefix abstraction;
5. benchmark harness;
6. MLP sanity run.

Exit criteria:

1. extracted prefixes match in-model prefix logits;
2. real latency/memory metrics are recorded;
3. all logs are reproducible from configs.

---

### Milestone 2: CIFAR proof

Deliverables:

1. CIFAR-10 and CIFAR-100 training runs;
2. ResNet-style PODS model;
3. naïve truncation baseline;
4. independent shallow baseline;
5. measured frontier plots.

Exit criteria:

1. PODS beats naïve truncation, or the current pass records an explicit objective-mismatch override with supporting analysis;
2. PODS is compared against independent shallow models;
3. at least three seeds for main variants.

Override note:

1. The current repository pass may accept Milestone 2 under explicit user override when PODS and naïve truncation are treated as materially different optimisation objectives rather than a strict like-for-like gate.
2. Any such override must be recorded in milestone validation notes and must not be applied silently.

---

### Milestone 3: Transformer proof

Deliverables:

1. BERT-base PODS wrapper;
2. RoBERTa-base PODS wrapper;
3. GLUE splits;
4. shallow BERT baselines;
5. LayerDrop-style baseline.

Exit criteria:

1. at least three GLUE tasks complete;
2. extracted prefix benchmarking works;
3. no hidden-state shortcut is used in cost measurement.

---

### Milestone 4: Full empirical sweep

Deliverables:

1. six or more GLUE tasks;
2. CIFAR results;
3. ablations;
4. uncertainty estimates;
5. raw logs.

Exit criteria:

1. every headline claim has direct measurement evidence;
2. every selected depth was chosen on selection-validation only;
3. final held-out evaluation is locked.

---

### Milestone 5: Paper assembly

Deliverables:

1. LaTeX paper;
2. tables generated from raw logs;
3. figures generated from raw logs;
4. reproducibility checklist;
5. limitations section;
6. code and config release package.

Exit criteria:

1. paper makes no unsupported speedup claims;
2. paper acknowledges when shallow specialists win;
3. paper is robust to the “this is just early exit” objection;
4. paper is robust to the “why not train shallow models” objection.

---

## 26. Final Summary

This project should be framed as **static prefix deployment**, not dynamic early exit.

The method trains one model so that many prefixes are valid standalone models. After training, each prefix is evaluated using real validation quality and real hardware measurements. The selected deployment model is a fixed shallow prefix.

The project will only hold up to top-tier review if it:

1. uses validation/test metrics rather than training accuracy;
2. benchmarks real extracted prefixes;
3. compares against independently trained shallow models;
4. compares against LayerDrop/OFA-style elastic baselines;
5. reports p50/p95 latency, throughput, memory, and file size;
6. reports uncertainty over seeds/examples;
7. makes a narrow claim and includes negative results if the method fails.

The strongest possible paper is not “we can remove layers.” That is known. The strongest possible paper is:

> A carefully measured study showing when ground-truth-supervised prefix-deployable training can produce a competitive static deployment frontier from one training run, and when existing shallow or elastic baselines remain better.
