# WORK LOG — MSc Thesis

**Ashwani Singh · Student ID 1229692**
MSc Machine Learning & Artificial Intelligence, Liverpool John Moores University (with upGrad)
Supervisor: Dr. Sitanath Biswas

**Thesis:** A Controlled Equal-Budget Comparison of GAN and Diffusion Data
Augmentation for Imbalanced Skin-Lesion Classification with a Memorization Audit

**How to use this file.** Append a new entry at the end of every working
session. Record what was done, what was decided, what was verified, and what
was left open. Commit it to git with the rest of the project — it is the
evidence behind the Research Plan chapter, and the record you refer back to when
something needs explaining months later.

**Phase mapping.** Keep the proposal's six-phase language (Table 9.1 and Figure
9.1), so the work log always lines up with the approved Gantt chart.

| Phase | Weeks | Activities |
|---|---|---|
| 1 | 1–2 | Literature review; dataset; lesion-level split; baseline classifier |
| 2 | 2–3 | Classical augmentation reference condition |
| 3 | 3–6 | Train adversarial generator and fine-tune latent diffusion within budget |
| 4 | 6–8 | Synthesise; rebalance; retrain and evaluate every condition |
| 5 | 8–10 | Memorization audit; quantity ablation; repeat seeds |
| 6 | 10–12 | Analysis; figures and tables; dissertation writing |

---

## Day 1 — Sunday 13 September 2026

**Phase:** 1 (setup and baseline data)
**Session:** approx. 14:50 – 21:00
**Machine:** local laptop (RTX 4060, i9-14900HX, Windows 11)

### Objective
Working environment, complete dataset verified, shared source modules in place.
No GPU training — deliberately, since the lesion-level split is the frozen frame
every later result depends on and is built once.

### Completed

**Accounts**
- Kaggle account; phone verification completed (required for any GPU access;
  Kaggle is the pre-registered PRIMARY execution environment, Run 1)
- Harvard Dataverse account created; HAM10000 terms of use accepted. This is the
  licence acceptance committed to in proposal §9.1
- (Pending) GitHub private repository `thesis-skin-lesion`

**Dataset acquisition**
- Source: Harvard Dataverse, DOI 10.7910/DVN/DBW86T — the source cited in
  proposal §7.2 and Table 8.1
- Downloaded: `HAM10000_images_part_1.zip` (1.3 GB),
  `HAM10000_images_part_2.zip` (1.3 GB), `HAM10000_metadata`
- Metadata downloaded without a file extension; renamed to
  `HAM10000_metadata.csv`. Confirmed genuinely comma-separated. Columns:
  `lesion_id, image_id, dx, dx_type, age, sex, localization, dataset`
- Both archives extracted into a single `images` folder
- **Verified image count: 10015** — exactly the published figure
- Segmentation masks and ISIC-2018 task files deliberately not downloaded; they
  are for a segmentation task and outside this study's scope

**Storage location decision**
- Project root set to `D:\thesis` (391 GB free)
- **Rejected: OneDrive.** Files On-Demand converts files to cloud placeholders,
  so a dataloader reading 10,015 images per epoch would hit intermittent stalls
  and read errors partway through a budgeted run. OneDrive also locks files
  during sync and corrupts git repositories. Proposal drafts remain in OneDrive;
  dataset, code and repository are on local disk

**Environment**
- Python 3.11.9 installed alongside existing 3.13.3; NOT added to PATH, so the
  system Python stays unambiguous. 3.11 chosen because PyTorch, diffusers and
  peft all ship reliable wheels for it
- "Disable path length limit" enabled during installation (260-character
  MAX_PATH would otherwise be reached by nested output paths)
- Virtual environment `D:\thesis\.venv` created with `py -3.11 -m venv`
- PowerShell execution policy set to `RemoteSigned` for CurrentUser, to permit
  venv activation
- **PyTorch 2.6.0+cu124** installed from the PyTorch index (NOT the default PyPI
  index, which serves a CPU-only build)
- Verified: `torch.cuda.is_available()` → `True`;
  `torch.cuda.get_device_name(0)` → `NVIDIA GeForce RTX 4060 Laptop GPU`
- Installed: pandas, matplotlib, seaborn, scikit-learn, scipy, tqdm, pyyaml;
  jupyter, ipykernel; diffusers, transformers, peft, accelerate, lpips,
  pytorch-fid
- All three import groups verified clean. No numpy 2.x conflict, despite
  numpy 2.4.6 arriving with PyTorch
- Jupyter kernel registered as "Python (thesis)"
- `requirements.txt` written via `pip freeze` — for the code appendix

**Note for implementation:** PyTorch 2.6 deprecates `torch.cuda.amp.autocast`
and `torch.cuda.amp.GradScaler`. Use `torch.amp.autocast('cuda')` and
`torch.amp.GradScaler('cuda')`. The Build Manual's code blocks still use the old
form and must be updated as each is implemented.

**Project structure created**

```
D:\thesis\
  .venv\            config\           src\
  data\raw\HAM10000\images\  (10,015 images)
  data\splits\      data\synthetic\
  notebooks\        results\          logs\
```

**Source modules placed in `src\`**

| File | Role |
|---|---|
| `config.yaml` | Single source of truth for every hyper-parameter |
| `data.py` | Recursive path index; dual label spaces; matched generator loaders |
| `budget.py` | Resumable, device-pinned GPU-hour enforcement; D9 checkpoints |
| `checkpointing.py` | Crash-safe training state; deterministic resumable sampling |
| `run_registry.py` | Keeps run1_cloud and run2_local separate and auditable |
| `metrics.py` | Equations 7.6–7.10, macro AUROC, bootstrap intervals |
| `__init__.py` | Marks `src` as a package |

Also downloaded, not yet in use: `gpu_probe.py` (hardware characterisation),
`sampling_pilot.py` (measures seconds-per-image before scheduling Run 1).

**Verified after placement**
- `from src.data import CLASSES, AUG_CLASSES` →
  `['akiec','bcc','bkl','df','mel','nv','vasc']` (7, classifier) and
  `['akiec','bcc','df','mel','vasc']` (5, generators)
- `config.yaml` → budget unit 4.0 GPU-hours; image path
  `D:/thesis/data/raw/HAM10000/images`

**Exploratory data analysis (`notebooks/01_eda.ipynb`)**

Published image counts confirmed exactly: 10,015 images across 7,470 unique
lesions. The 2,545 difference is duplicate photographs of the same physical
lesion, which is what forces lesion-level partitioning (Kaufman et al., 2012).

| Class | Images | **Unique lesions** | Images per lesion | Test lesions @15% |
|---|---|---|---|---|
| nv | 6,705 | 5,403 | 1.24 | 810 |
| bkl | 1,099 | 727 | 1.51 | 109 |
| mel | 1,113 | **614** | **1.81** | **92** |
| bcc | 514 | 327 | 1.57 | 49 |
| akiec | 327 | 228 | 1.43 | 34 |
| vasc | 142 | **98** | 1.45 | **15** |
| df | 115 | **73** | 1.58 | **11** |

Two findings worth carrying forward.

**Melanoma has the highest duplication rate** in the dataset, at 1.81 images
per lesion. Clinicians photographed suspicious lesions more often. An
image-level split would therefore leak most severely on precisely the class
named as the primary outcome — which strengthens the justification for
lesion-level partitioning (D11) in the methodology chapter.

**DF has only 73 unique lesions**, not 115 as the image count suggests, and
VASC only 98.

**D13 — confirmed as measured rather than estimated.** The decision log stated
DF would have "roughly 17 test images". At 11 test lesions and 1.58 images per
lesion this comes to approximately 17 images, so the estimate holds — but it is
now a measured figure. One reclassified image moves DF recall by roughly six
percentage points. VASC: 15 test lesions, approximately 21 images.

**D7 — FLAGGED FOR DECISION.** The synthetic target of 1,000 per class produces
wildly unequal synthetic-to-real ratios once measured against the actual
training partition:

| Class | Est. training images (~70%) | Target | Synthetic : real |
|---|---|---|---|
| mel | ~779 | 1,000 | 0.3 : 1 |
| bcc | ~360 | 1,000 | 1.8 : 1 |
| akiec | ~229 | 1,000 | 3.4 : 1 |
| vasc | ~99 | 1,000 | 9.1 : 1 |
| **df** | **~80** | 1,000 | **11.5 : 1** |

A 38-fold spread between MEL and DF. Generating 920 novel dermatofibroma images
from roughly 80 originals (themselves from only ~51 distinct lesions) makes
repetition close to certain, and substantially widens the ratio/source-size
confound already declared in D10.

This is not an error: D7 states explicitly that *"Exact ratios to be confirmed
from the measured training split before the target is fixed."* This is that
moment. The choice is between holding the target flat at 1,000 (unequal ratios,
simple to describe) or setting it as a multiple of each class's original count
(equal ratios, but unequal absolute quantities). **Decide after the split is
built and exact training counts are known — not before.**

### Decisions carried in from planning
Six defects were found in the Build Manual during planning and are corrected in
the modules now in place: the budget unit was inverted (D1); interval
checkpointing was absent (D9); the Kaggle image path reached only half the
dataset; the two generative arms would have trained on different source data
(D14); Run 1 did not fit the weekly Kaggle quota (D16); and AUROC was declared
but unused (D15). D17 (checkpoint-write time charged to the budget, scheduled by
wall clock, measured) was added while making training resumable.

### Open items
- [ ] GitHub private repository not yet created
- [ ] `.gitignore` not yet written — must exist before the first commit, so raw
      images are never committed (licence: no redistribution)
- [ ] `gpu_probe.py` not yet run (needs mains power, ~5 min soak)
- [ ] PRE_REGISTRATION §6 still states the efficiency curve costs 5–7 GPU-hours;
      the corrected estimate must be written in **before** `Date fixed:` is
      completed and the file committed
- [ ] **D7 open: synthetic target to be fixed once measured training counts are
      known.** Flat 1,000 per class, or a multiple of each class's original?
- [ ] `utils.py`, `split.py`, `classifier.py`, `gan.py`, `diffusion.py` not yet
      written
- [ ] Class-distribution figure (Figure 1 of the results chapter) not yet saved
      at 300 DPI

### Next session
1. Write `utils.py` (seed fixing) and `split.py` (lesion-level stratified split,
   aggregate-then-stratify per D11)
2. Run the split; confirm all four leakage assertions pass and class proportions
   hold across partitions to within ~2 percentage points
3. Read off the **exact** training-partition counts per class, and settle D7
4. Create the GitHub repository and `.gitignore`; correct PRE_REG §6; fill in
   `Date fixed:`; commit the frozen split and record the git hash

---

## Day 2 — Monday 21 September 2026

**Phase:** 1 (dataset and lesion-level split)
**Machine:** local laptop (RTX 4060)

### Objective
Write the seed and split modules, build the lesion-level split, and settle the
synthetic target (D7) against measured training counts.

### Completed

**Environment re-verified** after a gap of several days:
`torch 2.6.0+cu124`, `torch.cuda.is_available()` → `True`.

**`src/utils.py` — `set_seed()`**
Fixes Python, NumPy, PyTorch CPU and PyTorch CUDA generators, and sets cuDNN to
deterministic mode. Verified: two draws of `torch.rand(3)` after `set_seed(42)`
both returned `[0.8823, 0.9150, 0.3829]`.

**`src/split.py` — `lesion_level_split()` and `check_split()`**
Aggregate to one row per lesion → stratified 70/30 split on `dx` → split the 30
into 15/15 → map every image back to its lesion's partition. No image is
discarded. An added assertion verifies (rather than assumes) that no lesion
carries more than one diagnosis.

**Split run (seed 42) — all four leakage checks passed**

| Partition | Images | Lesions |
|---|---|---|
| train | 6,981 | 5,229 |
| val | 1,532 | 1,120 |
| test | 1,502 | 1,121 |
| **Total** | **10,015** | **7,470** |

Lesions divide exactly 70.0 / 15.0 / 15.0. Images divide 69.7 / 15.3 / 15.0,
because images-per-lesion varies.

**Image counts per class and partition**

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| akiec | 222 | 53 | 52 | 327 |
| bcc | 361 | 82 | 71 | 514 |
| bkl | 772 | 160 | 167 | 1,099 |
| df | 71 | 24 | 20 | 115 |
| mel | 773 | 173 | 167 | 1,113 |
| nv | 4,683 | 1,018 | 1,004 | 6,705 |
| vasc | 99 | 22 | 21 | 142 |

**Class proportions (image level)** agree across partitions to within
**0.68 percentage points** at most (BKL). Melanoma — the primary outcome — varies
by only 0.22 points. Stratification operates on lesions, so lesion proportions
are exact by construction; image proportions drift slightly, most visibly for DF
(61.7 / 20.9 / 17.4 of its images) because a few validation-set DF lesions carry
more photographs. DF lesions themselves divide ~51 / 11 / 11.

**Test support now measured, not estimated:** DF 20 test images, VASC 21.

**D18 recorded:** synthetic target confirmed at a flat 1,000 per class on the
measured counts. DF: 71 training images from 51 lesions, ratio 13.08 synthetic
per real image. Options B (fixed multiple) and C (ratio cap) considered and
rejected as headline; B declared exploratory in advance. See
`DECISION_LOG_ADDENDUM`.

**Decision log defect found and fixed:** the "Amendment required to
PRE_REGISTRATION_2.md" heading had been lost when D17 was inserted. Restored.

### Concepts clarified this session
- `lesion_id` identifies a **lesion, not a patient**. The dataset carries no
  patient identifier, so two lesions from one person can fall in different
  partitions. **Limitation to state:** lesion-level splitting removes
  same-lesion leakage but cannot remove patient-level correlation (shared skin
  tone, age, imaging conditions). Standard practice for HAM10000, but it must be
  stated.
- Leakage arises through pixels, not through `lesion_id`: the model never sees
  the identifier, but two photographs of one lesion are near-identical.
- A split percentage (how much data) is unrelated to an epoch (one full pass
  over the training data).
- Proportions can be read column-wise (share of each partition) or row-wise
  (share of each class sent to each partition). Both check stratification.
- The model never chooses a synthetic quantity at deployment; synthetic images
  exist only during training. The test set is kept real and imbalanced so that
  it reflects clinical prevalence.

### Problems and resolutions
- Earlier planning estimates of DF training images (~80) were optimistic; the
  measured figure is 71. Resolved by measurement before fixing the target, as D7
  required.

### Open items
- [ ] **Split CSVs NOT yet written to `data/splits/`.** The split exists only in
      notebook memory. Re-run with seed 42 (identical by construction) and save
      `train.csv`, `val.csv`, `test.csv` before anything else
- [ ] GitHub private repository and `.gitignore`
- [ ] Correct PRE_REG §6; fill in `Date fixed:`; commit the frozen split; record
      the git hash
- [ ] `gpu_probe.py` not yet run
- [ ] Class-distribution figure at 300 DPI not yet saved
- [ ] Ablation class not yet chosen (deferred to before the ablation; see D18)
- [ ] `classifier.py`, `gan.py`, `diffusion.py` not yet written

### Next session
1. Re-run the split (seed 42) and write the three CSVs to `data/splits/`
2. Verify the saved CSVs reproduce today's counts exactly
3. Create the GitHub repository and `.gitignore`
4. Correct PRE_REG §6, fill in `Date fixed:`, and commit the freeze

---

## Entry template

```markdown
## Day N — <Weekday> <DD Month YYYY>

**Phase:** <1–6>
**Session:** <start> – <end>
**Machine:** <local RTX 4060 / Kaggle P100 / Kaggle T4>

### Objective
<what this session was meant to achieve>

### Completed
<what was actually done, with verified outputs pasted in>

### Decisions
<any decision taken; assign a D-number and add it to DECISION_LOG.md>

### Problems and resolutions
<errors hit, what caused them, how they were fixed — the most useful section
to your future self>

### Budget consumed
<GPU-hours used this session, per arm, and on which device>

### Open items
- [ ] <carried forward>

### Next session
<the first thing to do next time>
```
