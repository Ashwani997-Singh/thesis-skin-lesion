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

---

## D20 — Sampling and throughput measured; no pre-registered quantity changes required

**Decision.** The sampling pilot required by D16 was run on the Run 1 device.
Together with a dataloader throughput measurement it establishes that Run 1
fits within two weeks of Kaggle quota **without** reducing any pre-registered
quantity. Therefore:

- The sampler remains **30 inference steps** with the default SD 1.5 scheduler
  (PNDM), as declared in `config.yaml`. The DPM-Solver++ fallback permitted by
  `PRE_REGISTRATION.md` §9 is **not** invoked.
- The efficiency curve remains **four checkpoints** (1.0 / 2.0 / 3.0 / 4.0
  GPU-hours) at **500 synthetic images per class**, as declared in D16.
- `num_workers = 4` is used for dataloaders **on Kaggle only**; Run 2 retains
  `num_workers = 0` because Windows Jupyter hangs with worker processes.

**Measured sampling rate** (Tesla T4, SD 1.5, fp16, 512 px, batch 8, guidance
7.5, 20 timed images after warm-up):

| Scheduler | Steps | s/image | Sampling total (15,000 images) |
|---|---|---|---|
| **PNDM (default)** | **30** | **4.82** | **20.1 GPU-h** |
| DPM-Solver++ | 25 | 4.40 | 18.3 GPU-h |
| DPM-Solver++ | 20 | 3.64 | 15.2 GPU-h |

Peak VRAM 6.47 GB. Note the saving is not proportional to the step reduction —
a third fewer steps gives only a quarter less time — because VAE decoding and
pipeline overhead are fixed per image.

**Measured classifier throughput** (ResNet-50, 128 px, batch 32, AMP):

| | Compute per step | Per epoch (compute) | Per epoch (measured) | Per run (~20 epochs) |
|---|---|---|---|---|
| num_workers=0 | 56.6 ms | 12.4 s | 89.0 s | 29.7 min |
| num_workers=2 | 56.6 ms | 12.4 s | 46.0 s | 15.3 min |
| **num_workers=4** | 56.6 ms | 12.4 s | **24.1 s** | **8.0 min** |

**The classifier was never GPU-bound.** At `num_workers=0`, 86% of each epoch
is spent decoding JPEGs while the GPU idles. Four worker processes reduce run
time 3.7-fold at no cost to the result — the data reaching the model is
identical, only its delivery changes.

**Revised Run 1 budget:**

| Item | GPU-hours |
|---|---|
| GAN training | 4.0 |
| Diffusion training | 4.0 |
| Headline sampling (5,000 images) | 6.7 |
| Efficiency-curve sampling (10,000 images) | 13.4 |
| 26 classifier runs at 8 min | 3.5 |
| FID and memorization audit | ~1.0 |
| **Total** | **~32.6** |

Split across two quota weeks — training and sampling in week one (~28 h),
classifier runs and audit in week two (~5 h) — this fits 60 GPU-hours with
substantial margin. Kaggle quota resets weekly; a single generative arm is
never split across weeks, because `BudgetTracker` would refuse to resume onto a
differently allocated device.

**Why no change was needed.** Earlier estimates put the classifier runs at
26–39 GPU-hours, which would have forced a reduction somewhere. That estimate
was wrong because it extrapolated from compute time alone. Measuring the
dataloader showed the true bottleneck and removed it. **The pilot's value was
not in licensing a reduction but in showing that none was required.**

**Scheduler reasoning, recorded for completeness.** Switching to DPM-Solver++
at 20 steps would have saved 4.9 GPU-hours, but it would have altered **every
synthetic image in the study**, including the headline conditions from which
all four research questions are answered. Sampler choice affects image quality
and diversity, and therefore bears directly on FID and the memorization audit.
A change of that reach was not worth 4.9 hours once the budget was shown to
fit. Had a reduction been unavoidable, reducing the curve — a supporting
analysis — would have been preferred to altering the headline images.

**A note on D16's stated preference.** D16 said that if quota were tight,
curve *quantity* should be reduced rather than intervals. Had a reduction been
necessary, that preference would have been reversed: at 250 synthetic images
per class, the addition to melanoma — which already has 773 real training
images — is a 32% increase, too small for any effect to be distinguishable
from the seed-to-seed variance measured in Condition A (melanoma recall
± 0.051). The curve would have been uninterpretable. Reducing to three
checkpoints at 500 images each would have preserved interpretability at every
point. This is recorded because the preference was formed before the sampling
rate was known; no action follows from it.

**Operational notes.**
- The Stable Diffusion safety checker is disabled (`safety_checker=None`).
  Dermatoscopic images trigger the NSFW filter, which would silently discard
  valid samples. No generated image is exposed publicly. To be stated in the
  methodology chapter.
- SD 1.5 weights are 4.27 GB and cached per session; the download recurs on
  each new Kaggle session.

**Proposal check.** Section 7.3, verbatim: *"The training budget will be
declared before experimentation and expressed in graphics processing unit
hours."* Sampling is outside the declared training budget (D2); these figures
are scheduling measurements, not budget changes. `PRE_REGISTRATION.md` §6
requires the pilot before Run 1; it has been run and recorded.

**Status: ALIGNED.** Fulfils the pilot required by D16 and confirms every
pre-registered quantity unchanged.

---

## D21 — Headline sampling from the end-of-budget checkpoint of each arm; observed artefacts reported, not corrected

**Decision.** Headline sampling (D7, D18) uses `gan_final.pt` for Condition C and
`lora_final` for Condition D. The generator architecture and every sampling
setting declared in `config.yaml` (30 inference steps, default PNDM scheduler,
guidance scale 7.5) remain unchanged after inspection.

**Why the final checkpoint.** It is the model the declared budget bought: the
state at which `BudgetTracker` recorded `gpu_hours_exhausted`. The 4.0 h
checkpoint is written as the threshold is crossed and the final checkpoint
moments later, so the two are near-identical; choosing one rule and applying it
to both arms matters more than which of the two is chosen.

**Inspection before sampling.** Both arms were inspected with one identical
procedure (`inspect_generators.py`, commit `46bcb11`): 32 samples per class per
arm, passed through the same 8-bit 128 px pathway the classifier receives, and
compared against 32 real training images per class. Run on the local RTX 4060
from Run 1 weights; inspection only, outside the budget (D2), and not used in
any reported result.

| | GAN | Diffusion |
|---|---|---|
| Spread ratio to real (pixel space) | 0.76 – 0.87 | 0.54 – 0.76 |
| Near-duplicate pairs | none | none |
| Label effect / noise effect | 0.76 | 0.70 |
| Mean colour vs real | within ~10 per channel | R and G about 30 lower |

LoRA adapter verified: 1.59M parameters loaded, max |lora_B| = 0.042, mean
pixel change against base SD 1.5 at identical seeds 43–53 (0–255 scale). Base
SD 1.5 without the adapter produces no dermatoscopic images at all.

**Artefacts observed, and why they are not corrected.** Each arm carries a
signature absent from real images: the GAN a checkerboard texture, the known
product of stride-2 transposed convolution (Odena, Dumoulin and Olah, 2016);
the diffusion arm a systematic darkening and shift toward magenta, consistent
with a high classifier-free guidance scale. Changing the architecture or the
guidance scale after observing outputs would be post-hoc tuning of one arm, and
retraining would break the equal-budget control. Both artefacts are reported as
findings and examined as possible classifier shortcuts in the discussion.

**Pixel-space caveat.** The spread ratio is a coarse indicator only: the GAN's
checkerboard itself inflates per-pixel variance, and the diffusion arm's uniform
background deflates it. Diversity is assessed properly by FID and LPIPS (Phase 5).

**Proposal check.** Section 7.3, verbatim: *"Where a model has not converged
within its budget, that outcome is itself a finding and will be reported rather
than corrected by additional training."* Table 9.1, Phase 3 contingency:
*"non-convergence reported as a finding"*. Sampling from the end-of-budget state,
and reporting rather than correcting what that state produces, applies the same
principle to sample quality.

**Status: ALIGNED.** Specifies a quantity the pre-registration leaves open; no
declared setting changed.

---

## D22 — Real-to-real calibration of the memorization threshold excludes images of the same lesion

**Decision.** When computing the real-to-real nearest-neighbour distribution
that calibrates the near-duplicate threshold (`PRE_REGISTRATION.md` §7,
Equation 7.5), a real image's nearest neighbour is searched among real images
of **other lesions only**. Images sharing its `lesion_id` are excluded. The
reference set is the generator source partition (1,526 images), the images
both generators were trained on.

**Why.** HAM10000 contains several photographs of many lesions — 62.5% of
melanoma lesions have more than one (Figure 4). If a second photograph of the
same lesion were allowed as the nearest neighbour, the real-to-real distribution
would be dominated by near-duplicates of a single physical lesion, its 1st
percentile would fall close to zero, and the threshold would become so strict
that genuine copying by a generator could pass undetected. The calibration
must answer "how close are two *different* lesions?", because that is the
distance below which a synthetic image stops being a new lesion.

**Direction of the risk.** This guards mainly against false negatives. The
diffusion arm's colour shift (D21) acts in the same direction, increasing the
distance of an otherwise copied image, so the closest synthetic–real pairs are
additionally inspected visually (`audit.n_worst_pairs_to_plot: 10`).

**Fixed before the audit.** Recorded before `src/memorization.py` was written
and before any synthetic-to-real distance was computed.

**Still to decide before the audit runs:** whether the threshold is pooled
across the five classes or set per class, and the fixed class for the
controlled sub-experiment (`PRE_REGISTRATION.md` §7, §9).

**Proposal check.** Section 7.5, verbatim: *"The threshold will be calibrated
using the distribution of distances between distinct real training images, so
that it reflects the natural similarity within the dataset rather than an
arbitrary cut-off."* Section 7.3 establishes that HAM10000 *"contains multiple
images of the same lesion"*. Two photographs of one lesion are not distinct in
the sense the calibration requires; D22 reads "distinct real training images"
as images of distinct lesions, consistent with the lesion-level treatment
applied to the split (D11).

**Consequence for data handling.** Section 9.1 states that *"any synthetic
images identified as near-duplicates of training images will not be published
or shared."* Flagged images are therefore excluded from any figure or release.

**Status: ALIGNED.** Specifies how the declared calibration is computed; the
pre-registration fixes the percentile but not the treatment of same-lesion pairs.

---

## D23 — Per-image predictions saved; Conditions A and B re-executed with C and D in one session; supplementary metrics and paired comparison fixed in advance

**Decision.** Recorded before any Condition C or D classifier has been trained.

1. Every classifier evaluation saves, next to its metrics JSON, the per-image
   true label, predicted label and class probabilities on the test partition
   (`<name>_predictions.npz`; numbers only, no image data). `src/metrics.py`
   changed accordingly (patch `patch_metrics_d23.py`).
2. Conditions A, B, C and D, seeds 42/43/44, are run in **one Kaggle session**,
   on one software environment and one code version. These re-executed A and B
   runs are the ones used in every comparison with C and D, **whatever their
   values**. The A and B runs of 2 October 2026 are retained unchanged and
   reported as a reproducibility check against the re-execution.
3. Supplementary metrics are added for all four conditions: macro precision,
   per-class specificity, and macro one-vs-rest AUPRC. Primary outcomes (D13)
   and model selection on validation macro-F1 are unchanged.
4. Paired comparison, fixed now. For each pair of conditions compared (D vs C
   for RQ1; C vs B and D vs B for RQ3) and each seed-matched pair of runs, the
   difference in macro-F1 and in melanoma recall is computed on the same
   bootstrap resamples of the test set (1,000 resamples, 95% percentile
   interval). A difference is called **consistent** when all three seed-matched
   intervals exclude zero in the same direction. McNemar's test on melanoma
   test images (correct vs incorrect) is reported per seed pair. Following
   `PRE_REGISTRATION.md` §8, p-values are descriptive only.

**Why.** The Second Marker asked for statistical testing to determine whether
the difference between GAN and diffusion augmentation is meaningful, for an aim
framed around statistically significant improvement, and for macro precision,
per-class specificity and AUPRC. The metrics saved so far are summaries; a
paired comparison on the same test images and AUPRC both require per-image
outputs, which were not stored. RQ3 compares the generative arms with
Condition B, so B needs per-image outputs as well, which is only possible by
re-executing it.

A second reason, not raised by the markers: A and B ran on 2 October under
PyTorch 2.10, C and D will run under 2.11 (Kaggle image updated between
sessions). Running all four conditions together removes that difference.

**Why the re-executed runs, decided now.** Choosing between the old and new A/B
runs after seeing C and D would be a post-hoc choice. The rule is fixed here:
the re-executed runs are used for all comparisons; the original runs serve
only as a reproducibility check.

**Cost.** About 1.6 GPU-hours for twelve classifier runs (8 min each, D20).

**Proposal check.** Section 7.6: each configuration is *"repeated across several
random seeds and reported as a mean with its spread, so that a genuine
difference between conditions can be distinguished from ordinary run-to-run
variation."* The paired comparison strengthens exactly this aim.

**Status: ALIGNED (extension in response to proposal feedback).** No primary
outcome, threshold or quantity changes. Adds supplementary measures and a
paired analysis fixed before the relevant results exist.

---

## D24 — Memorization audit: one pooled threshold, reference set fixed, DF as the controlled-sub-experiment class, and what that sub-experiment can show

**Decision.** Recorded before `src/memorization.py` is written and before any
synthetic-to-real distance is computed. Resolves the two items left open in D22.

1. **Distance and space.** LPIPS (AlexNet), as declared in `config.yaml`
   (`audit.feature_extractor: lpips_alex`), at 128 px, the resolution the
   classifier receives. Real images use the identical resize.
2. **Reference set.** The generator source partition (1,526 training images,
   five classes): the images both generators were trained on (D14). The nearest
   neighbour of a synthetic image is searched over **all** 1,526, not only its
   own class, because a copy is a copy whatever label it carries.
3. **One pooled threshold.** For each of the 1,526 real images, its nearest
   neighbour among real images of **other lesions** (D22) in the same reference
   set; the threshold is the 1st percentile (`audit.percentile: 1.0`) of these
   1,526 distances, pooled across classes.
4. **Per-class reporting without per-class thresholds.** The full distribution
   of nearest-neighbour distances is reported per class and per arm, alongside
   the near-duplicate rate under the single threshold. The ten closest
   synthetic-real pairs per arm are inspected visually (`n_worst_pairs_to_plot`).
5. **Controlled sub-experiment class: DF**, at 100, 250, 500, 1,000 and 2,000
   synthetic images per arm. Smaller sets are nested prefixes of the larger
   ones (image indices 0 to n-1, seeded per D2/D7 sampling), so every point uses
   the same images plus more. DF images 1,000 to 1,999 are generated for this
   purpose from the same final checkpoints and seeding scheme.

**Why one pooled threshold.** Proposal Section 7.5 specifies *"a near-duplicate
threshold"* calibrated on *"distances between distinct real training images"*,
and Table 7.x defines the near-duplicate rate as the share *"below the
calibrated threshold"*: one threshold throughout. A per-class 1st percentile
would also be unstable for DF and VASC: with 71 DF images it rests on roughly
one distance. The proposal's Phase 5 contingency (*"full distributions reported
rather than a single figure"*) is met by item 4.

**Why DF.** A class-conditional generator that learns DF from 71 images (51
lesions) has the narrowest real distribution to reproduce, so copying is most
plausible there; and any copy forms the largest share of that class's training
data, since DF has the fewest real images. MEL (773 source images) was
considered first and rejected: it is the primary classification outcome, but
the sub-experiment concerns memorization, and MEL is the class where copying is
least likely to be detectable. With a pooled threshold, DF's small size no
longer weakens the calibration.

**What the sub-experiment can and cannot show.** Drawing more samples does not
change the generator. The expected near-duplicate *rate* is therefore the same
at every quantity; differences between points reflect sampling variation, and a
rate that moved systematically would indicate an error. What does grow with
quantity is the **number** of near-duplicates and their **share of the DF
training data** the classifier sees (near-duplicates / (71 + n)). These two
quantities, together with downstream performance at each point (RQ4), are the
outcomes of the sub-experiment. The determinant of memorization itself, the
source-set size a generator is trained on, can only be varied by retraining the
generators, which the declared reporting language already names as future work
(`PRE_REGISTRATION.md` §7).

**Proposal check.** Section 7.5 as quoted above; Equation 7.5 defines
d(s) = min over x_i in X_train of ||phi(s) - phi(x_i)||, with X_train the images
the generators were trained on.

**Status: ALIGNED.** Specifies items `PRE_REGISTRATION.md` §9 lists as still
open (the sub-experiment class) and how the declared calibration is computed.
No threshold, percentile or quantity changes. Section 4 above clarifies the
interpretation of the declared sub-experiment rather than altering it.

---

## D25 — FID computation fixed, and generation of the extra DF images for the controlled sub-experiment

**Decision.** Recorded before any FID value or audit distance is computed.

1. **FID (proposal §7.6), `src/fid.py`.** Inception-v3 pool3 features through
   the reference `pytorch-fid` implementation. Real reference per class = the
   generator source images of that class (training partition, D14); pooled
   reference = all 1,526. Synthetic: all 1,000 per class per arm, 5,000 pooled
   per arm, so every comparison between arms uses equal counts. All images pass
   through the 128 px pathway the classifier receives before Inception's own
   resize to 299, identically for both arms. No validation or test image is
   used.
2. **Stated limitations.** Inception features come from ImageNet, so FID is
   indicative for dermatoscopy, not authoritative. FID is biased upward when a
   covariance is estimated from few images: with 71 real DF images (99 VASC) the
   per-class absolute values are unreliable, while the GAN-versus-diffusion
   comparison stays valid because both arms share the same reference and count.
   KID would be better behaved at small n; it is not added, because the proposal
   names FID, and is noted as future work.
3. **Extra DF images for D24.** DF images with indices 1,000 to 1,999 are
   generated per arm from the same final checkpoints (D21), the same sampler
   settings (30 steps, guidance 7.5, PNDM) and the same per-image seeding
   (`image_seed(42, "df", i)`), using the resumable generator so indices 0 to
   999 are reused, not regenerated. Before generating, five existing DF images
   per arm are regenerated and compared with the stored files, to confirm the
   seeding reproduces them. Sampling cost is reported separately (D2).

**Why.** FID with unequal sample counts, or with a reference that differs
between arms, would compare the estimator's bias rather than the generators.
Reusing the stored images and seeds makes the DF sets genuinely nested, as D24
requires.

**Status: ALIGNED.** Specifies how two declared measures are computed; no
declared quantity changes.

---

## D26 — Efficiency curve: three seeds per point, shared noise across checkpoints, and execution in three sessions

**Decision.** Recorded before any efficiency-curve image or classifier exists.

1. **Points and quantity, as declared.** Generator checkpoints at 1.0, 2.0, 3.0
   and 4.0 GPU-hours (D9), 500 synthetic images per class (D16), five augmented
   classes, both arms, added to the real training partition without classical
   augmentation (as in C and D). Same ResNet-50, training settings and test set
   as the headline conditions (`PRE_REGISTRATION.md` §6).
2. **Seeds: 42, 43 and 44 at every point** - 24 classifier runs (4 points x 2
   arms x 3 seeds), not the 8 single-seed runs used in the earlier cost
   estimate.
3. **Shared noise.** Curve images use the same per-image seeding as the
   headline set, `image_seed(42, class, i)` for i = 0..499. Image i of a class
   therefore starts from the same noise at every checkpoint, so differences
   along the curve reflect the generator's training, not the draw.
4. **Execution.** Sampling in two sessions (part a: 1.0 h and 2.0 h; part b:
   3.0 h and 4.0 h; about 7 GPU-hours each), each stored as its own private
   Kaggle dataset; then one session for the 24 classifier runs, saving
   per-image predictions as in D23. Sampling cost is reported separately (D2).
5. **Reading the curve.** Mean and seed spread of macro-F1 and melanoma recall
   at each point, per arm. The neutral framing in `PRE_REGISTRATION.md` §6 is
   kept: no directional prediction. The 4.0 h point is a separate draw of 500
   per class from the 4.0 h checkpoint, not the headline set, and is not
   compared with Conditions C and D as if it were.

**Why three seeds.** `PRE_REGISTRATION.md` §8 fixes seeds 42, 43 and 44 as the
minimum for reported results; the eight-run figure was a cost estimate, not a
design. Observed seed spread makes one seed insufficient: the re-executed
Condition A has a macro-F1 standard deviation of 0.048 across seeds, larger than
any plausible step between adjacent curve points. A single-seed curve could not
distinguish a trend from seed noise. Quota allows it: about 4.5 GPU-hours for
24 runs at roughly 11 minutes each.

**Why it matters for the feedback.** The First Marker's central criticism is
that a fixed budget may cut generators off before convergence. The curve is the
direct evidence: whether downstream benefit is still rising, flat, or falling
at 4.0 GPU-hours, for each arm.

**Status: ALIGNED.** Applies the declared seed minimum to a declared analysis;
no checkpoint, quantity or classifier setting changes.

---

## D27 — Quantity ablation (RQ4): DF only, nested quantities, Condition A as the zero point, three seeds

**Decision.** Recorded before any ablation classifier is trained.

1. **Class: DF.** Synthetic DF images are added at 100, 250, 500, 1,000 and
   2,000 per arm; **no other class receives synthetic images**. Every other
   setting is that of Conditions C and D: same real training partition, no
   classical augmentation, same ResNet-50, training settings and test set.
2. **Zero point = Condition A.** With no synthetic images and no augmentation
   the training set is exactly Condition A's, so the three re-executed A runs
   of 27 October (D23) are reused as quantity 0 rather than trained again.
3. **Nested sets, shared with the audit.** Quantity n uses DF images 0 to n-1
   of the final checkpoint of each arm: the same images, with the same seeds,
   whose nearest-neighbour distances were measured in the D24 sub-experiment.
   Images 0-999 are the stored headline set; 1,000-1,999 are regenerated with
   the same seeding, after the first five stored images are regenerated and
   compared (identical in the D25 check: mean absolute difference 0.0).
4. **Seeds 42, 43, 44** (`PRE_REGISTRATION.md` §8): 5 quantities x 2 arms x
   3 seeds = 30 runs, per-image predictions saved (D23).
5. **Outcomes.** DF recall and DF precision (does more synthetic DF raise
   over-prediction of DF?), macro-F1 and melanoma recall, as mean and seed
   spread at each quantity, with per-run bootstrap intervals. Reported next to
   the audit counts for the same images (D24), so the share of near-duplicates
   in the DF training data and the downstream effect are read together.

**Why DF, and why only DF.** The audit sub-experiment (D24) already uses DF, and
pairing the memorization audit with downstream benefit on the same images is the
study's second stated contribution. DF has the fewest real training images (71),
so it is where added synthetic data can plausibly change performance. Adding
synthetic images to one class only means any change is attributable to that
class's synthetic data (DECISION_LOG_ADDENDUM, relationship to the quantity
ablation). No new sampling from new checkpoints is needed.

**Limitation stated in advance.** DF has 20 test images, so one image moves DF
recall by five points. DF recall is therefore reported with bootstrap intervals
and read together with macro-F1; no claim rests on a single-point difference.

**Proposal check.** Objective 7: *"the effect of synthetic dataset size on
classification performance"*. Quantities as fixed in `config.yaml`
(`ablation_quantities: [0, 100, 250, 500, 1000, 2000]`).

**Status: ALIGNED.** Fixes the class left open in the analysis plan; quantities
and seeds as declared.

---

## D28 — Hyperparameter policy: no search in either arm; where each setting comes from, checked against its source

**Recorded:** 9 November 2026, after all Run 1 results. Records the origin of
settings that were fixed in `config/config.yaml` before generator training
(4 October 2026) and used unchanged in every session. **No value is changed by
this decision**, and no code is re-run.

**Why it is recorded now.** The First Marker asked how a hyperparameter search
fits within the fixed budget (F5). The planned answer was "no search in either
arm; published defaults in both". A check of `config.yaml` against the source
papers on 9 November showed that the GAN optimiser settings are not the
defaults of one paper. The wording is therefore corrected to what was actually
used, rather than the settings being changed after the results.

**Policy.** No hyperparameter search was run in either arm. A search consumes
compute, so searching within the 4.0 GPU-hour budget would leave less training
time, and searching outside it would break the equal budget (Lucic et al.,
2018). Each arm instead uses standard, untuned settings from the literature or
the reference implementation, fixed before training.

**Settings and their sources.**

| Arm | Setting | Used | Source value | Source |
|---|---|---|---|---|
| GAN | Critic updates per generator update | 5 | 5 | Gulrajani et al. (2017), Algorithm 1 |
| GAN | Gradient-penalty weight λ | 10 | 10 | Gulrajani et al. (2017), Algorithm 1 |
| GAN | Adam β2 | 0.9 | 0.9 | Gulrajani et al. (2017), Algorithm 1 |
| GAN | Adam learning rate | 0.0002 | 0.0001 (Algorithm 1 default); 0.0002 (their CIFAR-10 ResNet experiment) | Gulrajani et al. (2017), Algorithm 1 and Appendix F; Radford, Metz and Chintala (2015) |
| GAN | Adam β1 | 0.5 | 0 (Gulrajani Algorithm 1); 0.5 (DCGAN) | Radford, Metz and Chintala (2015) |
| GAN | Batch size | 64 | 64 | Gulrajani et al. (2017), Appendix F |
| Diffusion | LoRA learning rate (AdamW) | 0.0001 | 0.0001 | Hugging Face diffusers, `train_text_to_image_lora.py` |
| Diffusion | LoRA target modules | `to_k`, `to_q`, `to_v`, `to_out.0` | same | diffusers reference script |
| Diffusion | Batch size, gradient accumulation | 1, 8 | 1, 4 (example command) | diffusers LoRA guide |
| Diffusion | LoRA rank r | 8 | not prescribed; small ranks, including 8, are evaluated | Hu et al. (2022) |
| Diffusion | LoRA α | 16 (scale α/r = 2) | equal to r in the diffusers script (scale 1) | diffusers reference script |
| Diffusion | Sampler | PNDM, 30 steps, guidance 7.5 | SD 1.5 default scheduler and guidance; steps fixed in D20 | Rombach et al. (2022); D20 |
| Classifier | AdamW, learning rate 0.0001, weight decay 0.0001, cosine schedule, patience 7 | — | standard fine-tuning settings | Identical in all conditions, so not a source of difference between arms (D19) |

**What this means.**

1. **GAN.** The critic schedule, penalty weight and β2 are the WGAN-GP
   defaults. The learning rate (0.0002) matches the WGAN-GP authors' own
   CIFAR-10 setting and DCGAN; β1 = 0.5 is the DCGAN value rather than the
   WGAN-GP Algorithm 1 value of 0. The combination is not the default of a
   single paper, and is reported as such.
2. **Diffusion.** The learning rate and target modules are the reference
   implementation's. α = 16 with r = 8 doubles the adapter scale relative to
   the diffusers script (α = r). Gradient accumulation of 8 rather than 4
   gives an effective batch of 8.
3. **No setting was tuned, and none was changed after any result.** Both
   generators trained to 4.0 GPU-hours with the values above, as recorded in
   `budget_state_run1_cloud_*.json` and the `config.yaml` committed before
   training.

**Consequences for the conclusions.** Neither arm received a tuned
configuration, so the comparison is between two untuned, standard setups at an
equal budget. It does not show what either family achieves with tuned
settings; the β1 choice for the GAN and the α/r scale for LoRA are both
plausible sources of difference that are not tested. This is stated as a
limitation (Chapter 6), together with the existing scope limitation (S4):
conclusions hold for these instances, not for the GAN and diffusion families in
general.

**Not changed:** `config.yaml`, `src/`, every checkpoint and every result.

**Wording to use from now on:** "standard, untuned settings taken from the
WGAN-GP and DCGAN papers and from the diffusers LoRA reference implementation,
fixed before training" — not "published defaults".

**Status: ALIGNED.** Clarifies the provenance of settings already fixed;
answers part of F5 and S4; adds one limitation.
