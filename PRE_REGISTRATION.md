# Pre-registration of analysis plan

**Student:** Ashwani Singh (1229692)
**Supervisor:** Dr. Sitanath Biswas
**Thesis:** A Controlled Equal-Budget Comparison of GAN and Diffusion Data Augmentation
for Imbalanced Skin-Lesion Classification with a Memorization Audit

**Date fixed:** 23 September 2026

**Frozen split commit:** `0e2add07fc53bd9a62760b187c4301f1165917fe`
(Wed 23 Sep 2026 23:33:52 +0400) — the lesion-level partition, configuration and
source modules were committed to version control before any generative model
existed.

**Supersedes** the drafts dated 22 August 2026, which declared optimiser steps
as the primary budget unit. That inverted the unit declared in Section 7.3 of the
approved proposal and was corrected in D1. Decisions D14–D18 are incorporated
here; see `DECISION_LOG.md` for the full reasoning on each.

---

## 1. Two execution environments

The full protocol is executed twice. Within each execution both generative arms run on
identical hardware, so each execution is an internally valid equal-budget comparison.
Results from the two executions are reported separately and are **never pooled**.

| | Environment | Status | Rationale |
|---|---|---|---|
| **Run 1** | Cloud GPU, Kaggle free tier | **PRIMARY** | Matches the hardware specification in Table 8.2 of the approved proposal |
| **Run 2** | Local laptop, NVIDIA RTX 4060 (8 GB), i9-14900HX, 31.7 GB RAM, Windows 11 | **SECONDARY (replication)** | Hardware became available after proposal submission; guarantees both arms on one fixed device |

This designation is fixed in advance and **will not be revised after results are observed**.
If the two executions disagree, both are reported and the disagreement is discussed as a
finding about hardware-dependence of budget-constrained conclusions.

Resuming a generative arm onto a different GPU model is refused in code
(`BudgetTracker` raises `DeviceChanged`), because equal GPU-hours on unequal
hardware is not an equal budget.

## 2. Outcomes, declared in advance

Test support below is **measured** from the frozen split, not estimated.

| Outcome | Status | Test images | Justification |
|---|---|---|---|
| Melanoma recall | Primary | 167 | Stable support; one image moves recall by 0.6 pp |
| Macro-F1 | Primary | — | Weights all classes equally (Eq 7.9) |
| Balanced accuracy | Supporting | — | Imbalance-aware counterpart to accuracy (Eq 7.10) |
| BCC recall | Supporting | 71 | Moderate support; 1.4 pp per image |
| AKIEC recall | Supporting | 52 | Moderate support; 1.9 pp per image |
| VASC recall | Secondary | 21 | Low support; 4.8 pp per image |
| DF recall | Secondary | 20 | Low support; 5.0 pp per image |
| Macro AUROC (one-vs-rest) | Supplementary | — | Threshold-independent view; never used for model selection (D15) |
| Overall accuracy | Reported only | — | Never used for model selection or as a headline |
| FID | Image quality | — | Equal sample counts across conditions |
| Near-duplicate rate | Privacy | — | Threshold calibrated against real-to-real distances |

## 3. Budget

**Declared unit: 4.0 GPU-hours per generative arm.**
Section 7.3 of the approved proposal states that the budget "will be declared before
experimentation and expressed in graphics processing unit hours". GPU-hours is therefore
the declared and authoritative unit, and this pre-registration does not change that (D1).

**Additionally logged: optimiser steps** (cap 20,000 per arm), recorded as a
hardware-invariant cross-check. This is supplementary logging, not a redefinition. It is
recorded because a laptop GPU reduces its clocks as it heats, so four wall-clock hours late
in a run represent less computation than four hours at the start. Where the two units
disagree, both are reported and the discrepancy is discussed.

- Both arms in an execution must report the same device string (`assert_same_device`)
- Cumulative GPU-hours persist across interrupted sessions; a resumed arm does not
  receive a fresh budget (`budget_state_*.json`)
- GPU temperature and clocks are logged during Run 2 so throttling can be reported honestly
- Non-convergence within budget is reported as a finding, not corrected with extra training
  (proposal 7.3: "Where a model has not converged within its budget, that outcome is itself
  a finding and will be reported rather than corrected by additional training")

### 3.1 Scope of the budget — DECIDED
The declared budget covers **generator training only**. Image sampling (generation) is
**excluded** and reported separately, because sampling cost differs by roughly two orders
of magnitude between the families (one forward pass per image for the GAN; 30 denoising
steps at 512 px for diffusion). Including it would consume diffusion training time for a
reason unrelated to learning quality. The proposal wording ("the training budget") supports
this reading. Sampling wall-clock time is reported as a practical observation (D2).

Pretraining is likewise outside the budget, which covers **adaptation to HAM10000 only**.
Pretrained latent-diffusion weights are treated as a freely available resource of the same
status as the ImageNet weights used by the classifier. Trainable-parameter counts are
reported for both arms so the asymmetry is visible rather than hidden.

### 3.2 Checkpoint-writing overhead — DECIDED (D17)
Time spent writing checkpoints is **charged to the declared budget**; the clock is not
paused. Resume checkpoints are written on a **wall-clock interval of 10 minutes of
cumulative budget**, not on a fixed number of optimiser steps, because the two arms run at
very different step rates and a step-based interval would give the faster arm several times
as many saves. Cumulative checkpoint-writing time is measured and reported for both arms.

## 4. Conditions — DECIDED

| Condition | Training data |
|---|---|
| A Baseline | Original data. **Standard resizing and ImageNet normalisation only; no augmentation or balancing.** |
| B Classical | Original + geometric/photometric augmentation |
| C GAN | Original + conditional WGAN-GP synthetic images |
| D Diffusion | Original + LoRA latent-diffusion synthetic images |

**Classical augmentation is NOT applied to conditions C and D.** Each condition therefore
adds exactly one thing to the baseline, giving clean attribution and making RQ3 a direct
C-vs-B and D-vs-B comparison. If time permits, an optional Condition E (classical +
diffusion) tests whether the two stack; this is exploratory and declared as such.

Classifier: ResNet-50, ImageNet-pretrained, **identical across all conditions and all
checkpoint evaluations**. Model selection on validation macro-F1, never accuracy.

### 4.1 Generative source data — DECIDED (D14)
Both generative arms train on **one shared source partition**: the training split
restricted to the five augmented classes (AKIEC, BCC, DF, MEL, VASC). "Identical source
data" means the same images, the same split and the same class subset; the arms differ
**only** in resize (128 px for the GAN, 512 px for diffusion, per D6). This is asserted in
code (`assert_matched_sources`). Restricting to five rather than extending to seven
concentrates a fixed budget on the classes actually synthesised; training on all seven
would spend roughly two thirds of each arm's budget on melanocytic nevi, which neither arm
generates. The consequent limitation — that the from-scratch GAN is denied additional
images from which shared dermatoscopic structure might transfer — is accepted and stated.

## 5. Synthetic quantity — DECIDED, confirmed on measured counts (D18)

Headline experiment uses **target = 1,000 images per augmented class** (MEL, BCC, AKIEC,
VASC, DF). D7 required this to be confirmed against the measured training split before
being fixed. It has been confirmed rather than revised.

**Measured training-partition counts** (lesion-level split, seed 42):

| Class | Train images | Train lesions | Synthetic needed | Synthetic per real |
|---|---|---|---|---|
| mel | 773 | 430 | 227 | 0.29 |
| bcc | 361 | 229 | 639 | 1.77 |
| akiec | 222 | 159 | 778 | 3.50 |
| vasc | 99 | 69 | 901 | 9.10 |
| df | 71 | 51 | 929 | 13.08 |

The synthetic-to-real ratio spans a factor of 45 across the five classes and **is reported
in the results table**, because it is material to interpretation. Alternatives considered
and rejected: a fixed multiple of each class's original count (equalises the ratio but
barely reduces imbalance) and a ratio cap (introduces a threshold chosen after observing
the data). Option B may be run as an **exploratory** condition only if time remains after
all pre-registered work is complete; it is declared exploratory here, before any results
exist, so that it cannot later be promoted to the headline.

## 6. Efficiency curve — ADDED

Generator checkpoints are saved at **1.0 / 2.0 / 3.0 / 4.0 GPU-hours** (with the
corresponding optimiser-step count recorded at each).
Synthetic images from each checkpoint are used to train the downstream classifier, giving
macro-F1 as a function of cumulative GPU-hours. Framed neutrally in advance:

> "Downstream macro-F1 is plotted against cumulative GPU-hours to evaluate structural
> efficiency and trajectory, capturing whether performance has plateaued or remains on an
> upward trajectory at the pre-declared budget cut-off."

No directional prediction is made.

**Quantity at curve points: 500 synthetic images per class** (D16), against 1,000 for the
headline conditions. The curve measures trajectory across budget and remains internally
valid provided quantity is constant across all four checkpoints and both arms; the headline
conditions, from which the research questions are answered, are unaffected.

**Cost, corrected.** An earlier draft costed this item at 5–7 GPU-hours by counting only
the eight classifier runs and omitting the sampling those runs depend on. Each checkpoint
requires its own synthetic images: 4 checkpoints × 2 arms × 5 classes. At 500 per class
that is 10,000 diffusion images (GAN sampling is negligible by comparison), of the order of
**11 GPU-hours**, plus 5–7 for the classifier runs — approximately **16 GPU-hours per
execution**. A sampling timing pilot is run before Run 1 to replace this estimate with a
measured seconds-per-image figure, which is recorded in the decision log.

**The same fine-tuned ResNet-50 is used**, not a linear probe or lightweight head, so curve
points are comparable with the headline conditions. If quota is tight, reduce the curve
quantity further rather than changing the classifier or dropping intervals.

## 7. Memorization audit and the ratio confound — ADDED

Headline audit: near-duplicate rate per class for both arms, threshold calibrated against
the real-to-real nearest-neighbour distribution (Eq 7.5).

**Known confound, declared in advance:** across the five augmented classes, synthetic-to-real
ratio and source-set size covary (DF has both the highest ratio, 13.08, and the fewest
source images, 71 from 51 lesions). With five classes these cannot be separated, and
**no threshold will be claimed** from the headline experiment.

**Controlled sub-experiment:** the memorization audit is additionally run at each quantity
ablation point (100, 250, 500, 1,000, 2,000 synthetic images) for **one fixed class**,
holding source-set size constant while varying the ratio. This is the only analysis from
which any ratio-related claim will be made. The class used is fixed before the ablation is
run and recorded in the decision log.

Reporting language fixed in advance:
> "Because ratio and source-set size covary in this dataset, these factors cannot be
> separated across classes; establishing a threshold would require a designed experiment
> varying ratio at fixed source size, which is proposed as future work."

## 8. Seeds and uncertainty

Seeds 42, 43, 44 minimum (five if budget allows). Report mean and standard deviation across
seeds **and** bootstrap 95% confidence intervals over the test set. These measure different
things: seeds capture training variance, bootstrap captures small-test-set variance.
With three to five seeds any significance test has low power and p-values are treated as
descriptive only.

## 9. What may still change (must be recorded if it does)

- Resolution, if 512 px proves infeasible in 8 GB — **applied equally to both arms and both runs**
- Number of seeds, if time is short — applied equally to all conditions
- Curve quantity or number of checkpoint intervals, if quota is tight — applied equally to
  both arms
- Sampling scheduler and step count, if the timing pilot shows 30 steps is infeasible —
  applied identically to every sampling call in both executions
- Note: the 20,000-step figure is a cap, not a commitment; the GPU-hour budget governs
- Reduced replication in Run 2 (e.g. one seed, no ablation) — declared as partial
- The class used for the controlled memorization sub-experiment

## 10. What may not change

- Which execution is primary
- The primary outcomes
- The lesion-level split, frozen in git at commit `0e2add07` before any generator existed
- Exclusion of classical augmentation from conditions C and D
- Exclusion of sampling time and pretraining from the declared budget
- GPU-hours as the declared budget unit
- The headline synthetic target of 1,000 per class
