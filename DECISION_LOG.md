# DECISION LOG
Every decision taken during planning, why it was taken, and the verbatim proposal wording
it is checked against. All quotes below were extracted directly from the submitted PDF
(`Research_Proposal_LJMU_Final.pdf`) and verified present, not recalled from memory.

**Verification method:** text extracted from pages 6 onward of the submitted PDF and each
quoted phrase confirmed by exact string match. Items marked NOT IN PROPOSAL were searched
for and confirmed absent.

---

## D1 — Budget unit: GPU-hours declared, optimiser steps additionally logged

**Decision.** 4.0 GPU-hours per arm is the declared and authoritative budget unit.
Optimiser steps (target 20,000) are logged additionally as a cross-check.

**Why.** A laptop GPU throttles as it heats, so equal wall-clock does not guarantee equal
computation. Logging steps protects the comparison. But the proposal fixes the unit.

**Proposal check.** Section 7.3, verbatim: *"The training budget will be declared before
experimentation and expressed in graphics processing unit hours."*

**Status: CORRECTED.** An earlier draft of this plan made optimiser steps primary. That
inverted the declared unit and has been reversed. Logging more than promised is permitted;
redefining the declared unit is not.

---

## D2 — Budget covers generator training only; sampling excluded and reported separately

**Decision.** The 4.0 GPU-hours covers training the generator. Image generation
(sampling) is excluded from the budget and its cost reported separately.

**Why.** Sampling cost differs by roughly two orders of magnitude between the families:
one forward pass per image for the GAN, versus 30 denoising steps at 512 px for diffusion.
Charging sampling to the budget would consume diffusion *training* time for a reason
unrelated to learning quality, reintroducing an asymmetry the design exists to remove.

**Proposal check.** Objective 2: *"within a pre-declared **training** budget."* Objective 3:
*"within the same pre-declared **training** budget."* Section 7.3: *"The **training** budget
will be declared..."* The proposal consistently says training budget.

**Status: ALIGNED.** Sampling time was never inside the declared budget.

---

## D3 — Conditions C and D receive NO classical augmentation

**Decision.** Condition C is baseline + GAN images. Condition D is baseline + diffusion
images. Neither also receives geometric/photometric augmentation.

**Why.** Each condition then adds exactly one thing to the baseline, giving clean
attribution, and RQ3 becomes a direct C-vs-B and D-vs-B comparison. An optional exploratory
Condition E (classical + diffusion) may test whether the two stack, declared as exploratory.

**Proposal check.** Section 7.1, verbatim: *"The independent variable is the composition of
the training dataset, which takes four levels: unaugmented, classically augmented
(Shorten and Khoshgoftaar, 2019), adversarially augmented and diffusion augmented."*
Four parallel levels, not nested ones.

**Status: ALIGNED.**

---

## D4 — Condition A wording tightened

**Decision.** Condition A described as *"standard resizing and ImageNet normalisation only;
no augmentation or balancing applied"* rather than *"no augmentation"*.

**Why.** A ResNet requires resizing and normalisation to produce tensors at all. Calling
that "no augmentation" is imprecise and invites an easy examiner challenge.

**Proposal check.** Section 7.3, verbatim: *"Images will be resized to a common resolution
and normalised."* The proposal already states preprocessing occurs.

**Status: ALIGNED.** Terminology clarification, not a design change.

---

## D5 — Melanoma included among the generated classes

**Decision.** Five classes generated: MEL, BCC, AKIEC, VASC, DF.

**Why.** Melanoma recall is the declared primary outcome. If melanoma were not generated,
the headline metric could only improve as a side effect of rebalancing other classes.

**Proposal check.** Section 7.4, verbatim: *"Both generators will synthesise images for the
minority categories"* — generic, no class list. Section 7.6, verbatim: *"Recall on the
minority classes, and on melanoma in particular, is the primary outcome of this study."*
MEL at 11.1% against NV at 66.9% is plainly a minority category.

**Status: ALIGNED, and more faithful than excluding it.** The proposal never enumerates
classes; excluding melanoma would have been the deviation.

---

## D6 — Diffusion trained at 512 px, generated images downsampled to 128 px

**Decision.** Classifier and GAN operate at 128 px. LoRA diffusion trains at 512 px, its
outputs downsampled to 128 px for the classifier.

**Why.** The Stable Diffusion VAE downsamples by a factor of eight. At 128 px input the
latents would be 16x16 against the 64x64 the UNet expects — badly off-distribution. Training
there would produce poor samples and invite the false conclusion that diffusion underperforms,
when the true cause is misconfiguration.

**Proposal check.** No resolution number appears anywhere in the proposal (searched: "128"
occurs zero times). Scope, verbatim: *"Images are processed at a reduced resolution
appropriate to the declared training budget."* Table 9.1 Phase 3 contingency, verbatim:
*"reduced resolution available; rank and step counts reduced equally across both arms."*

**Status: ALIGNED.** Resolution was deliberately left unspecified. Fallback ladder
512 → 384 → 256, applied equally to both arms and both runs.

---

## D7 — Synthetic target 1,000 per class, with ratios reported

**Decision.** Headline experiment uses target = 1,000 per augmented class. The
synthetic-to-real ratio per class is reported in the results table.

**Why.** Full balancing to 6,705 would mean generating ~6,590 dermatofibroma images from
~80 training originals, which guarantees repetition and invites memorization. A moderate
target is defensible, and the quantity ablation tests it empirically rather than by assertion.
Ratios vary substantially across classes (MEL low, DF high) and are material to interpretation.

**Proposal check.** Section 7.4, verbatim: *"in quantities sufficient to reduce the imbalance
in the training partition."* No number specified. Objective 7 commits to *"analyse the effect
of synthetic dataset size on classification performance."*

**Status: ALIGNED.** Exact ratios to be confirmed from the measured training split before
the target is fixed; the figures used in planning were estimates from a 70% split.

---

## D8 — Two executions: cloud primary, local replication

**Decision.** Full protocol run twice. Run 1 on Kaggle GPU is the pre-registered primary,
matching Table 8.2 of the proposal. Run 2 on the RTX 4060 is the secondary replication.
Within each execution both arms use the same GPU. Results are never pooled.

**Why.** The proposal specifies a cloud-hosted GPU; a discrete local GPU became available
afterwards. Running both gives replication (stronger than a single run) while keeping the
primary result matched to the approved specification.

**Proposal check.** Table 8.2 specifies *"Cloud-hosted NVIDIA GPU with 16 GB video memory,
accessed through a free-tier notebook service"* and *"Local graphics: Integrated graphics —
visualisation only; not used for model training."*

**Status: PARTIAL DEVIATION, DISCLOSED.** Table 8.2 used a placeholder specification that
was never updated with the real machine. The dissertation must update the table to the
actual hardware (RTX 4060 8 GB, i9-14900HX, 31.7 GB RAM, Windows 11) and state that a
discrete GPU became available after submission and was used for the replication. Hardware
specification is a resourcing statement, not a research commitment; better hardware than
declared does not affect feasibility, but it must be stated rather than left silent.

---

## D9 — Efficiency curve added

**Decision.** Generator checkpoints saved at 1.0 / 2.0 / 3.0 / 4.0 GPU-hours. Synthetic
images from each are used to train the downstream classifier, giving macro-F1 against
cumulative GPU-hours for both arms.

**Why.** Answers "what does each method buy per unit of compute, and had either plateaued at
the cut-off?" without a second training regime. Considered and rejected: a separate
run-to-completion experiment, because GANs have no reliable convergence criterion (any
stopping rule would be contestable) and it would roughly double an already full workload.

**Framing fixed in advance, neutral:** *"Downstream macro-F1 is plotted against cumulative
GPU-hours to evaluate structural efficiency and trajectory, capturing whether performance has
plateaued or remains on an upward trajectory at the pre-declared budget cut-off."*
No directional prediction is made.

**Cost.** 8 extra classifier runs per execution (4 intervals x 2 arms), approximately
5–7 GPU-hours. **The same fine-tuned ResNet-50 is used**, not a linear probe, so curve points
remain comparable with the headline conditions.

**Proposal check.** Section 7.3, verbatim: *"the budget consumed will be logged and reported."*
Section 7.4, verbatim: *"The classifier used throughout is a residual convolutional network
(He et al., 2016) initialised from weights pretrained on ImageNet."* — confirms fine-tuning
throughout, so a linear probe would itself be the deviation.

**Status: ALIGNED.** Additional analysis of data already committed to be produced.

---

## D10 — Memorization audit at ablation points, and the ratio confound declared

**Decision.** Headline audit reports near-duplicate rate per class for both arms. Because
synthetic-to-real ratio and source-set size covary across the five augmented classes, **no
threshold claim is made from the headline experiment.** A controlled sub-experiment runs the
audit at each quantity-ablation point (100 / 250 / 500 / 1,000 / 2,000) for one fixed class,
holding source size constant while varying the ratio.

**Why.** DF has both the highest ratio and the fewest source images, so the two factors
cannot be separated with five data points. The ablation sub-experiment varies ratio at fixed
source size, converting a descriptive observation into a controlled test at near-zero extra
cost, since those synthetic images already exist.

**Reporting language fixed in advance:** *"Because ratio and source-set size covary in this
dataset, these factors cannot be separated across classes; establishing a threshold would
require a designed experiment varying ratio at fixed source size, which is proposed as
future work."*

**Proposal check.** Section 7.5, verbatim: *"The threshold will be calibrated using the
distribution of distances between distinct real training images, so that it reflects the
natural similarity within the dataset rather than an arbitrary cut-off."* Objective 7
commits to the quantity ablation.

**Status: ALIGNED.** Interpretation of results already committed to.

---

## D11 — Lesion-level stratified split retained over GroupKFold / GroupShuffleSplit

**Decision.** Aggregate to one row per lesion_id, perform a stratified split on that table,
then map lesion IDs back to images.

**Why.** `GroupKFold` and `GroupShuffleSplit` preserve grouping but do not stratify, so class
proportions would drift — material with roughly 90 DF lesions. `StratifiedGroupKFold` struggles
when a class has few groups because it must satisfy both constraints at once. Aggregating first
dissolves the conflict: each row is already its own group, so an ordinary stratified split
achieves grouping and stratification simultaneously.

**Proposal check.** Section 7.3, verbatim: *"partitioning will be performed at lesion level
rather than image level, ensuring that no lesion contributes images to more than one
partition."* Also: *"No synthetic image will enter the validation or test partitions under
any condition."*

**Status: ALIGNED.** State the rationale explicitly in the methodology chapter.

---

## D12 — Seeds and uncertainty reporting

**Decision.** Seeds 42, 43, 44 minimum. Report mean and standard deviation across seeds
**and** bootstrap 95% confidence intervals over the test set.

**Why.** These measure different things: seeds capture training variance, bootstrap captures
small-test-set variance. With three to five seeds any significance test has low power, so
p-values are treated as descriptive only.

**Proposal check.** Section 7.6, verbatim: *"Each configuration is repeated across several
random seeds and reported as a mean with its spread."*

**Status: ALIGNED, exceeds commitment.** Bootstrap intervals are additional rigour.

---

## D13 — Primary and secondary outcomes declared before results

**Decision.** Primary: melanoma recall, macro-F1. Secondary: DF and VASC recall, explicitly
labelled low-support. Overall accuracy reported but never used for model selection.

**Why.** DF has roughly 17 test images, so one reclassified image moves recall by about six
points. A gap between arms of that size on DF is one image, not a finding. MEL has 1,113
images and is stable.

**Proposal check.** Section 7.6, verbatim: *"Recall on the minority classes, and on melanoma
in particular, is the primary outcome of this study."* And: *"Overall accuracy is not an
appropriate primary measure on an imbalanced dataset."*

**Status: ALIGNED.**

---

## Summary of status

| # | Decision | Status |
|---|---|---|
| D1 | Budget unit GPU-hours, steps logged additionally | **Corrected** (was inverted) |
| D2 | Budget covers training only | Aligned |
| D3 | No classical augmentation on C and D | Aligned |
| D4 | Condition A wording tightened | Aligned |
| D5 | Melanoma included in generated classes | Aligned |
| D6 | Diffusion at 512 px | Aligned |
| D7 | Target 1,000, ratios reported | Aligned |
| D8 | Two executions, cloud primary | **Partial deviation, must be disclosed** |
| D9 | Efficiency curve | Aligned |
| D10 | Audit at ablation points, confound declared | Aligned |
| D11 | Lesion-level stratified split | Aligned |
| D12 | Seeds plus bootstrap intervals | Aligned |
| D13 | Primary/secondary outcomes declared | Aligned |

**Only two items require action:** D1 is now corrected in the pre-registration. D8 requires
one paragraph in the dissertation methodology and an updated Table 8.2.



# DECISION LOG — ADDENDUM (D14–D18)

Append to `DECISION_LOG.md`. Same convention: decision, reasoning, verbatim
proposal wording it was checked against, status.

D14 to D16 arose from cross-checking the Build Manual against
`PRE_REGISTRATION_2.md` and `DECISION_LOG.md` before any code was written.
D17 arose from making the training loops resumable across cloud sessions.
D18 confirms D7 against the measured training split.

---

## D14 — Both generative arms train on the same five-class source partition

**Decision.** The GAN and the LoRA diffusion model both train on the training
partition restricted to the five augmented classes (AKIEC, BCC, DF, MEL,
VASC). "Identical source data" means the same images, the same split, and the
same class subset; the arms differ **only** in resize (128 px for the GAN,
512 px for diffusion, per D6).

**Why.** As drafted, the GAN loader used the full seven-class training
partition (~7,010 images) while the diffusion loader was bounded by the
five-class prompt dictionary (~1,550 images). Under an equal-budget design
that is an uncontrolled asymmetry in the very factor the study exists to hold
constant.

Restricting both to five is preferred over extending both to seven because,
under a fixed four-hour budget, roughly two thirds of gradient steps on the
full partition would be dominated by melanocytic nevi — a class neither arm
generates. Concentrating a scarce budget on the classes actually synthesised
is the more faithful reading of an equal-*budget* comparison.

**Implementation.** `src/data.py` exposes a single `generator_source_frame()`
used to build both loaders, and `assert_matched_sources()` fails if the two
arms' `image_id` sets differ. The generators use their own five-class label
space (`AUG_CLASSES` / `AUG_C2I`), distinct from the classifier's seven-class
`CLASSES` / `C2I`. The GAN's conditional embedding is sized from
`AUG_CLASSES`; sizing it at seven while feeding only five labels would leave
two embedding rows untrained and cause labels passed at generation time to
index the wrong row.

**Limitation to disclose.** Excluding NV and BKL denies the from-scratch GAN
additional images from which shared dermatoscopic structure (texture,
vignetting, hair artefacts) might transfer. The adversarial arm may therefore
be operating below its best achievable configuration on this dataset. This is
accepted deliberately: a matched comparison under an unequal-but-declared
configuration is more defensible than an unmatched one, and the constraint is
applied identically to both arms. To be stated in the limitations section.

**Proposal check.** Section 7.4, verbatim: *"Both generators will synthesise
images for the minority categories."* Section 7.1, verbatim: *"All factors
other than the training data, including the classifier architecture, the
optimiser (Kingma and Ba, 2015), the number of training epochs, the random
seeds and the test partition, are held constant across conditions so that
observed differences can be attributed to the augmentation strategy."*
The proposal never enumerates the generative training source.

**Status: ALIGNED.** Removes an asymmetry the proposal's own control
principle forbids.

---

## D15 — Macro one-vs-rest AUROC reported as a supplementary metric

**Decision.** Macro OvR AUROC is computed for every condition and reported in
the results tables as a supplementary measure. It never drives model
selection (validation macro-F1 continues to do that, per the proposal) and is
never presented as a headline outcome.

**Why.** AUROC appears in the proposal's List of Abbreviations but nowhere in
the body. The options were to delete the abbreviation or to use the measure.
Using it costs three lines in `src/metrics.py`, converts a consistency gap
into an additional reported metric, and adds a threshold-independent view
alongside the threshold-dependent recall and F1 figures.

**Implementation.** `collect_predictions()` returns softmax probabilities
alongside argmax predictions. `macro_auroc()` wraps `roc_auc_score` in a
`try/except ValueError` returning NaN: the function raises when a class is
absent from `y_true`, which cannot occur on the full test partition but does
occur on bootstrap resamples of DF and VASC.

**Proposal check.** List of Abbreviations, verbatim: *"AUROC Area Under the
Receiver Operating Characteristic Curve."* Section 7.6 fixes the primary
outcome and does not exclude supplementary measures.

**Status: ALIGNED, additive.** Resolves Handover §12 item 6.

---

## D16 — Synthetic quantity for efficiency-curve points, and a sampling pilot

**Decision.** Efficiency-curve checkpoints (D9) use **500 synthetic images
per class**, against 1,000 for the headline conditions (D7). The quantity is
identical across all four checkpoints and both arms. A sampling timing pilot
is run before Step 14 and the figure confirmed or revised.

**Why.** The efficiency curve requires its own synthetic images at every
checkpoint: four checkpoints × two arms × five classes. At 1,000 per class
that is 20,000 diffusion images at 512 px, which on a free-tier cloud GPU is
of the order of 17–28 GPU-hours for that item alone — larger than every
classifier run in the study combined, and beyond a two-week quota when added
to the rest of Run 1. `PRE_REGISTRATION_2.md` §6 costed this item at 5–7
GPU-hours by counting only the classifier runs and not the sampling they
depend on.

Reducing the curve quantity does not change what the curve measures. The
curve reports downstream macro-F1 as a function of cumulative GPU-hours, and
remains internally valid provided quantity is held constant across all curve
points and both arms. The headline conditions, which are what the research
questions are answered from, are unaffected at 1,000.

**Sampling pilot.** Before committing, generate 20 images at the declared
sampler settings, time them, and extrapolate. Record the measured rate. This
replaces an estimate with a measurement and produces the sampling-cost figure
that D2 already commits to reporting separately.

**Scheduler.** The sampler remains as declared (30 inference steps, guidance
7.5, default SD1.5 scheduler). Switching to a faster solver is recorded as a
declared fallback only, to be applied identically to every sampling call in
both executions if the pilot shows the reduced quantity is still infeasible.

**Proposal check.** Section 7.4, verbatim: *"in quantities sufficient to
reduce the imbalance in the training partition"* — no number specified.
Objective 7 commits to analysing *"the effect of synthetic dataset size on
classification performance"*, which the quantity ablation (unchanged at
0/100/250/500/1,000/2,000) delivers.

**Status: ALIGNED.** Permitted explicitly by `PRE_REGISTRATION_2.md` §9
("Number of seeds or checkpoint intervals, if time is short — applied equally
to all conditions"). Update §6 of the pre-registration to reflect the
corrected cost estimate and the 500 figure **before** the date is filled in
and the file is committed.

---

## D17 — Checkpoint-writing time is charged to the budget, scheduled by wall clock, and measured

**Decision.** Three parts.

1. Time spent writing checkpoints is **counted inside** the declared 4.0
   GPU-hours. The clock is not paused during saves.
2. Resume checkpoints are written on a **wall-clock interval of 10 minutes of
   cumulative budget**, not on a fixed number of optimiser steps. Both arms
   therefore write 24 resume checkpoints. This is separate from, and additional
   to, the four D9 efficiency-curve checkpoints at 1.0/2.0/3.0/4.0 GPU-hours,
   which are also charged to the budget.
3. Cumulative checkpoint-writing time is **instrumented and reported** as an
   absolute figure and as a share of the declared budget, for both arms.

**Why count rather than exclude.** Pausing the budget clock during saves would
be marginally more accurate, but it adds branching to the one mechanism the
central contribution depends on. A bug that pauses the clock incorrectly, or
fails to resume it, corrupts the equal-budget claim invisibly. Counting the
time is simpler, cannot fail silently, and — being measured — can be shown to
be small rather than merely claimed to be.

**Why a wall-clock interval and not a step interval.** The two arms run at very
different step rates: the GAN completes on the order of 20,000 optimiser steps
in four hours, the diffusion arm on the order of 5,000 (batch size 1, gradient
accumulation 8, 512 px). A save every 1,000 steps would give the GAN roughly
four times as many saves as diffusion. Its resume checkpoint is also larger
(generator, critic and two Adam states, ~90 MB, against a LoRA adapter and its
optimiser state at ~20 MB), so the faster arm would pay more per save *and*
save more often. A policy intended as symmetric would be asymmetric in the
declared unit. Scheduling by wall clock makes the save count identical and the
overhead near-identical, and bounds crash-recovery loss uniformly at ten
minutes for both arms rather than at three minutes for one and forty-eight for
the other.

**Expected magnitude.** 24 resume saves plus 4 curve checkpoints, at roughly
one second each, is of the order of 30 seconds against 14,400 — approximately
0.2% of the budget, and closely matched between arms. The measured figure is
reported rather than this estimate.

**Implementation.** `BudgetTracker.save_due(every_minutes=10.0)` fires once per
interval of cumulative elapsed budget; the next mark is recomputed from
cumulative time rather than stored, so a resumed session neither re-fires nor
skips a save. `BudgetTracker.timed_save()` is a context manager wrapping the
write and accumulating its cost into `checkpoint_write_seconds`, which is
persisted in the budget state file alongside
`checkpoint_write_share_of_budget`.

**Proposal check.** Section 7.3, verbatim: *"The training budget will be
declared before experimentation and expressed in graphics processing unit
hours."* And: *"the budget consumed will be logged and reported."* Charging
save time to the declared unit is the literal reading; excluding it would
require an argument the proposal does not make. Section 7.1 requires factors
other than the training data to be held constant across conditions, which a
step-scheduled interval would violate in wall-clock terms.

**Status: ALIGNED.** Additional instrumentation of a quantity the proposal
already commits to logging.

---

## D18 — Synthetic target confirmed at a flat 1,000 per class, on measured counts

**Decision.** The headline synthetic target remains **1,000 images per
augmented class** (AKIEC, BCC, DF, MEL, VASC), as declared in D7. The decision
was re-examined once the lesion-level split had been built and exact
training-partition counts were known, as D7 required, and was confirmed rather
than revised.

**Measured training-partition counts** (lesion-level split, seed 42):

| Class | Train images | Train lesions | Synthetic needed | Synthetic per real image |
|---|---|---|---|---|
| mel | 773 | 430 | 227 | 0.29 |
| bcc | 361 | 229 | 639 | 1.77 |
| akiec | 222 | 159 | 778 | 3.50 |
| vasc | 99 | 69 | 901 | 9.10 |
| df | 71 | 51 | 929 | 13.08 |

The planning estimates had been ~80 training images and a ratio of ~11.5 for
DF. The measured position is tighter: 71 images from 51 distinct lesions, and a
ratio of 13.08. Across the five classes the ratio spans a factor of 45.

**Options considered.**

- **A — Flat 1,000 per class (adopted).** Reduces imbalance most. Ratios vary
  from 0.29 to 13.08.
- **B — A fixed multiple of each class's original count (e.g. 3×).** Equalises
  the ratio across classes, but leaves DF at 284 images against roughly 4,700
  NV, so imbalance is barely reduced; and requires 2,319 synthetic MEL images.
  Rejected as the headline because it fails the stated purpose of augmentation.
- **C — 1,000 with a ratio cap (e.g. 5×).** A compromise, but introduces a new
  threshold chosen after the counts were observed, with no principled basis for
  its value. Rejected.

**Why A.**

1. *The proposal anticipated this regime.* The high DF and VASC ratios are not
   an unforeseen difficulty; the proposal identifies these classes as precisely
   where generative augmentation is most informative and memorization most
   likely. The memorization audit is designed for this regime.
2. *A is the declared position.* B and C would each introduce a quantity chosen
   after observing the data. Retaining the pre-registered target where it
   remains defensible avoids any appearance of post-hoc adjustment.
3. *A serves the purpose of augmentation.* Only A substantially reduces the
   imbalance in the training partition.

**Consequences, accepted and stated in advance.** The DF and VASC arms
extrapolate far beyond their source data, so downstream results for those
classes may be weak and near-duplicate rates may be high. Either outcome is a
finding, not a failure. The ratio/source-size confound across classes, already
declared in D10, is correspondingly wide; no threshold claim is made from the
headline experiment.

**Relationship to the quantity ablation.** Option A does not answer how the
quantity of synthetic data affects performance; the quantity ablation (RQ4,
Objective 7) does, by varying quantity for one class at fixed source size. A
change from A to B alters all five classes at once and cannot attribute an
effect to any one of them, so the ablation is the cleaner test of the
underlying question. **The class used for the ablation is not yet fixed** and
will be decided before the ablation is run. DF would address the scarcest
regime directly but carries the noisiest test support (20 images).

**Exploratory option.** Option B may be run as an additional exploratory
condition only if time remains after all pre-registered work is complete. It is
declared exploratory now, before any results exist, so that it cannot later be
promoted to the headline on the strength of its outcome. Because synthetic
images are generated with a deterministic per-image seed (`image_seed` in
`src/checkpointing.py`), images already generated for A are reused, and B would
require only the additional MEL and BCC images plus six classifier runs per
execution.

**Proposal check.** Section 7.4, verbatim: *"Both generators will synthesise
images for the minority categories in quantities sufficient to reduce the
imbalance in the training partition."* Section 7.2, verbatim: *"Dermatofibroma
and vascular lesions are represented by roughly one hundred images each, which
is the regime in which generative augmentation is most likely to be informative
and also, following Carlini et al. (2023), most likely to memorise."* D7,
verbatim: *"Exact ratios to be confirmed from the measured training split before
the target is fixed."*

**Status: ALIGNED.** D7 confirmed on measured data. `PRE_REGISTRATION_2.md` §5
requires no change.

---

## Amendment required to PRE_REGISTRATION_2.md

Section 6 currently states the efficiency curve costs "approximately 5–7
GPU-hours". Replace with the corrected estimate including sampling, and state
the curve quantity. Make this edit **before** filling in `Date fixed:` and
committing at Step 8, so the committed pre-registration is the accurate one.

---

## D19 — Early-stopping criterion fixed at patience 7, not tuned

**Decision.** Checkpoint selection uses validation macro-F1 with an
early-stopping patience of **7 epochs** and a maximum of **30 epochs**. These
values were fixed in `config/config.yaml` and committed to version control in
commit `0e2add07` (23 September 2026, 23:33 +0400) — **before any training run**
— and applied identically to every condition, every seed and both generative
arms. They were **not tuned**, and are not revised in light of observed results.

**Why this entry exists.** Neither value appears in the approved proposal or in
`PRE_REGISTRATION.md`. Section 7.1 of the proposal requires "the number of
training epochs" to be held constant across conditions but specifies no figure,
and the pre-registration fixes only the selection *metric* ("model selection on
validation macro-F1, never accuracy"), not the stopping rule. The choice is
therefore recorded here so that its provenance and timing are documented rather
than implicit in a configuration file.

**Why these values.** Patience 2–3 would stop prematurely: validation macro-F1
is itself noisy on this dataset, since DF and VASC contribute only 24 and 22
validation images respectively, so a single favourable early epoch can start the
counter while the model is still improving. Patience 15 against a 30-epoch
ceiling would rarely trigger at all, making the mechanism inert and wasting GPU
time after improvement has stopped. Seven sits between the two. No claim is made
that it is optimal.

**Why it was not tuned.** Three reasons.

1. *The study's independent variable is the composition of the training data*
   (proposal §7.1), not classifier hyperparameters. The classifier is held
   constant by design; optimising it is outside the declared scope.
2. *Tuning would consume the budget the study exists to allocate elsewhere.*
   Evaluating five candidate values across four conditions and three seeds is
   sixty classifier runs, of the order of forty GPU-hours — compute committed to
   the generative comparison.
3. *Legitimate tuning would have required stricter conditions* than are
   available here: selection on validation data only, completed before any test
   evaluation, and a single value applied to all conditions. Selecting a value
   after observing test results would be exactly the practice pre-registration
   exists to prevent.

**Why a suboptimal value does not threaten the comparison.** The criterion is
applied identically to all four conditions, all seeds and both arms. If patience
7 stops training earlier than ideal, it does so equally everywhere, shifting
every condition's absolute level by a similar amount while leaving the
*differences between conditions* — which answer the research questions —
intact.

**Observed effect, recorded as a limitation.** In the Condition A baseline,
early stopping triggered at epochs 25, 9 and 20 for seeds 42, 43 and 44, with
best epochs 18, 2 and 13. Seed 43 stopped after ten epochs because a favourable
epoch-2 score started the patience counter early; its test macro-F1 (0.5909) is
the lowest of the three, and its AKIEC recall collapsed to 0.346 against 0.673
for seed 42. This contributes materially to the observed seed-to-seed variance
(melanoma recall 0.459 ± 0.051 across three seeds) and is stated in the
limitations section. It is a property of the criterion applied uniformly, not a
bias between conditions.

**Reporting language fixed in advance:**
> "Early stopping with patience 7 on validation macro-F1 triggered at different
> epochs across seeds. Validation macro-F1 is itself noisy on this dataset, since
> DF and VASC contribute only 24 and 22 validation images, so an early
> favourable epoch can begin the patience counter prematurely. The criterion was
> fixed before any training run and applied identically to all conditions,
> seeds and generative arms, so it does not bias the comparison, but it
> contributes to the reported seed-to-seed variance."

**Proposal check.** Section 7.1, verbatim: *"All factors other than the training
data, including the classifier architecture, the optimiser (Kingma and Ba,
2015), the number of training epochs, the random seeds and the test partition,
are held constant across conditions so that observed differences can be
attributed to the augmentation strategy."* The proposal requires constancy and
specifies no value; holding these constant satisfies it.

**Status: ALIGNED.** Documents a value the proposal deliberately left
unspecified, fixed before observation and applied uniformly.
