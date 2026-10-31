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

## Day 3 — Wednesday 23 September 2026

**Phase:** 1 (dataset, split, freeze)
**Machine:** local laptop (RTX 4060)

### Objective
Save and verify the lesion-level split, produce the exploratory figures, and
freeze the split in version control before any generative model exists.

### Completed

**Split reproduced and saved.** Re-ran `lesion_level_split(df, seed=42)` in a
fresh session and obtained numbers identical to Day 2 (train 6,981 / val 1,532 /
test 1,502 images; 5,229 / 1,120 / 1,121 lesions). This is the practical
demonstration that the seed makes the partition reproducible. Written to
`data/splits/{train,val,test}.csv` (488 KB, 107 KB, 105 KB).

**Saved files verified independently of memory:** re-read from disk, all three
leakage assertions pass, 10,015 images accounted for, no `synthetic` column
present, per-class counts match exactly.

The first two rows of `train.csv` happen to illustrate the whole rationale:
`HAM_0000118` appears twice, as `ISIC_0027419` and `ISIC_0025030` — one lesion,
two photographs, both in train. Under an image-level split these could have been
separated.

**Five figures produced, all verified at 300 DPI:**

| File | Content |
|---|---|
| `fig01_class_distribution.png` | Images per class; the five augmented classes highlighted |
| `fig02_images_vs_lesions.png` | Images against unique lesions, with the ratio per class |
| `fig03_photos_per_lesion.png` | Distribution of photographs per lesion |
| `fig04_duplication_by_class.png` | Share of lesions with more than one photograph, by class |
| `fig05_real_samples.png` | Four real training images per class, seeded for reproducibility |

**New finding — duplication is not uniform across classes.** Of 7,470 lesions,
1,956 (26.2%) carry more than one photograph, with a maximum of six.

| Class | Lesions | Mean photos | >1 photo |
|---|---|---|---|
| mel | 614 | 1.81 | **62.5%** |
| df | 73 | 1.58 | 46.6% |
| bcc | 327 | 1.57 | 46.5% |
| bkl | 727 | 1.51 | 39.5% |
| vasc | 98 | 1.45 | 34.7% |
| akiec | 228 | 1.43 | 33.8% |
| nv | 5,403 | 1.24 | **18.3%** |

Melanoma is the most heavily duplicated class and melanocytic nevi the least — a
three-fold difference. Clinicians photographed suspicious lesions more
thoroughly. The consequence matters: an image-level split would leak most
severely on precisely the class declared as the primary outcome. This turns D11
(lesion-level splitting) from a convention into measured evidence, and the
paragraph is drafted for the methodology chapter.

Note for the write-up: `max_photos` was examined and deliberately not reported,
because it scales with class size (NV reaches 6 simply because it has 5,403
lesions) and so measures sample size rather than duplication. The percentage is
reported instead.

**Version control established.**
- Git installed and on PATH (2.55.0); VS Code restarted to pick it up
- `.gitignore` written before the first commit: excludes `data/raw/`,
  `data/synthetic/`, checkpoints, `*.pt`, `.venv/`
- `git init -b main`; user name and email set locally (not globally, so the
  existing DevOps repository is unaffected)
- Staged and inspected before committing: exactly 20 files, no `.jpg`, no `.venv`
- **FREEZE commit `0e2add07fc53bd9a62760b187c4301f1165917fe`,
  Wed 23 Sep 2026 23:33:52 +0400** — the lesion-level split committed before any
  generative model exists. Record this hash in the methodology chapter.
- Private GitHub repository `Ashwani997-Singh/thesis-skin-lesion` created
  (no README, no .gitignore, no licence, to avoid unrelated histories)
- Pushed: 6.48 MiB — consistent with code, CSVs and figures only

**Pre-registration corrected and committed (`02d7af7`).** Three earlier drafts
existed in Downloads; the two dated 22 August still declared optimiser steps as
the primary budget unit, which D1 had corrected. Only the latest was carried
into the repository, renamed `PRE_REGISTRATION.md` so one authoritative version
exists. Seven corrections applied:

1. §6 cost: 5–7 GPU-hours was wrong — it counted the eight classifier runs but
   omitted the sampling those runs depend on. Corrected to ~16 GPU-hours
   (~11 sampling + 5–7 classifier), pending the timing pilot
2. §6 lever: "reduce to three intervals" replaced with reducing curve quantity,
   per D16
3. `Date fixed:` completed, with the freeze commit hash
4. §5: measured ratios substituted for estimates (MEL 0.29 to DF 13.08), per D18
5. §2: measured test support substituted for estimates (MEL 167, DF 20, VASC 21)
6. D14, D15 and D17 incorporated, which post-dated the 26 August draft
7. §10: commit hash and budget unit added to the list that may not change

### Problems and resolutions
- `git` not recognised in the VS Code terminal although installed at
  `C:\Program Files\Git\cmd\git.exe`. Cause: VS Code had been started before Git
  entered PATH, and a process keeps the PATH it started with. Fixed by fully
  restarting VS Code.
- `git config` failed with "not in a git directory" — `git init` had not been run
  yet. Resolved by initialising first.
- Figures were briefly saved twice: once by `plt.savefig()` at 300 DPI and once
  manually from the notebook at ~100 DPI. The manual copies were roughly a
  quarter the file size and would not have met the university's 300 DPI
  requirement. Removed. **Rule adopted: figures are written by `plt.savefig()`
  into `results/figures/` only; the notebook preview is not the artefact.**

### Proposal alignment reviewed
Checked whether the declared quantities constitute deviations. They do not: the
proposal deliberately specifies no figure for budget hours, resolution, synthetic
quantity, augmented classes or seed count, stating only that the budget "will be
declared before experimentation". Of eighteen logged decisions, sixteen are
ALIGNED, one is a CORRECTION to an internal draft (D1), and **one is a genuine
partial deviation (D8, hardware)** which requires a paragraph in the methodology
chapter and an updated Table 8.2.

### Open items
- [ ] Figure 6 (split composition across partitions) not yet produced
- [ ] `gpu_probe.py` and `sampling_pilot.py` still in Downloads, not yet run
- [ ] Decision-log addendum (D14–D18) not yet copied into the repository
- [ ] Class-level split-proportion table not yet written into the methodology draft
- [ ] Ablation class still to be fixed (before the ablation is run)
- [ ] `classifier.py`, `gan.py`, `diffusion.py` not yet written
- [ ] README not yet written for the repository

### Next session
1. Figure 6 — split composition, the visual evidence that stratification held
2. Move `gpu_probe.py` into the repository and run it (mains power, 5-minute soak)
3. Copy `DECISION_LOG.md` and the addendum into the repository and commit
4. Begin `classifier.py` — the Condition A baseline

---

## Day 4 — Thursday 24 September 2026

**Phase:** 1 (documentation, hardware characterisation)
**Machine:** local laptop (RTX 4060)

### Objective
Complete the remaining Phase 1 documentation, characterise the hardware before
any training, and bring every planning document into version control.

### Completed

**Figure 6 — split composition.** Class shares across the three partitions, on a
log y-axis (NV at 67% and DF at 1% cannot both be read on a linear scale). Within
each class the three bars sit at almost the same height, which is the visual
evidence that stratification held. Largest deviation 0.67 percentage points.
DF shows the only visible gap (train 1.02%, validation 1.57%): stratification
operated on lesions, and the validation DF lesions happen to carry more
photographs each. Commit `fdd87b8`.

**Hardware characterisation (`gpu_probe.py`, 5-minute soak, mains power).**

| Measurement | Result |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop, 8.59 GB, CC 8.9 |
| ResNet-50 @128 bs32 (AMP) | 36.5 ms/step, peak 0.87 GB |
| WGAN-GP @128 bs32 (fp32) | 535 ms/step, peak 1.05 GB → ~26,900 steps in 4 h |
| WGAN-GP @128 bs64 (fp32) | 1069 ms/step, peak **1.64 GB** → ~13,500 steps in 4 h |
| SM clock over 300 s | 2625 MHz, **drift +0.0%** |
| Temperature | stable 61–65 °C |
| Peak power | 78.3 W (hardware max 140 W) |

Two results matter more than expected.

**No thermal throttling at all.** The planning documents assumed a laptop GPU
would reduce clocks over a long run — `PRE_REGISTRATION.md` §3 justifies logging
optimiser steps partly on that basis. Over five minutes of sustained WGAN-GP
load the clock did not move, and the temperature settled well below the thermal
limit. Four GPU-hours therefore represents the same quantity of computation at
the end of a run as at the start, which is exactly what the equal-budget control
requires. The step logging remains as a cross-check, but the concern that
motivated it is weaker than assumed for this machine. To be re-measured over a
full four-hour run before this is stated as a finding.

**VRAM is not a constraint for the GAN arm.** Estimates during planning were
4–6 GB for WGAN-GP at batch 64; the measured figure is **1.64 GB** of 8.59 GB.
Batch 64 is comfortable, and roughly 6.5 GB remains free for the 512 px LoRA arm.

Note: batch 64 takes almost exactly twice the time of batch 32, and utilisation
sits at 98–100%, so the GPU is saturated rather than latency-bound. At batch 64,
four GPU-hours yields ~13,500 generator steps, below the 20,000 step cap — the
cap will therefore not bind, which is correct, since GPU-hours is the declared
unit.

**Repository completed.**
- `gpu_probe.py`, `sampling_pilot.py`, `gpu_probe_results.json` committed (`5ef3f42`)
- `DECISION_LOG.md` assembled: the original D1–D13 and the D14–D18 addendum
  concatenated into one file, verified at 18 entries (`20dbef8`). This was
  necessary because `PRE_REGISTRATION.md` references `DECISION_LOG.md` for the
  reasoning behind D14–D18, and that reference was broken while the addendum sat
  outside the repository
- `README.md` written: study description, measured dataset counts, repository
  structure, the freeze commit hash, environment reproduction steps, and the
  hardware characterisation (`e3eaa81`)
- `requirements.txt` generated with `pip freeze`

### Problems and resolutions
- `FileNotFoundError` on `data/raw/HAM10000/HAM10000_metadata.csv` when
  re-running the notebook in a new session. Cause: the kernel's working directory
  was `notebooks/`, not the project root. Fixed by adding
  `os.chdir(r"D:\thesis")` at the top of the first cell. **Adopt this as the
  first line of every notebook.**
- `gpu_probe.py` and `sampling_pilot.py` had not in fact been downloaded in an
  earlier session; `Move-Item` failed with PathNotFound. Re-obtained and moved.

### Concepts clarified this session
- A notebook stores cell text in the `.ipynb` file but variables live in the
  kernel process, so they are lost when the kernel stops. Data must be reloaded
  each session. From now on the split is **read from `data/splits/*.csv`**, never
  regenerated — the partition is frozen.
- Batch size is how many images the model processes before one weight update; a
  step is one such update. For WGAN-GP one generator step comprises five critic
  updates plus one generator update (`n_critic: 5`).
- **Where the models come from, and why they differ.** No pretrained WGAN-GP
  exists for dermatoscopic images, and adversarial generators do not transfer
  across domains as diffusion models do, so the GAN is implemented and trained
  from random initialisation. Stable Diffusion 1.5 is downloaded (~5 GB) and
  adapted with LoRA. A pretrained adversarial generator would also make the
  equal-budget control unmeasurable, since the compute already spent could not be
  accounted for. The diffusion arm's pretraining is admitted because it is freely
  and publicly available — the same assumption the study already makes for the
  ImageNet-pretrained classifier in all four conditions.
- **The claim this licenses.** Not "diffusion is the better architecture", which
  would not survive the asymmetry, but "under a fixed adaptation budget with
  freely available pretrained weights, which pipeline returns more minority-class
  recall". Trainable-parameter counts are reported for both arms.

### Proposal alignment
No new deviations. The hardware result strengthens D8: a dedicated local GPU with
no measurable throttling is a better guarantee of the equal-budget control than
free-tier cloud hardware, which is the argument the methodology chapter will make
when disclosing the Table 8.2 change.

### Open items
- [ ] `classifier.py` not yet written — Condition A baseline is the last
      remaining Phase 1 item
- [ ] `sampling_pilot.py` not yet run (needs Stable Diffusion weights; belongs to
      the Kaggle session)
- [ ] Ablation class still to be fixed, before the ablation is run
- [ ] Split CSVs not yet uploaded to Kaggle as a private dataset
- [ ] Kaggle notebook not yet created; Run 1 not yet begun
- [ ] `gan.py`, `diffusion.py` not yet written

### Next session
1. Write `src/classifier.py` — ResNet-50, AdamW, cosine schedule, early stopping,
   **checkpoint selection on validation macro-F1, never accuracy**
2. Run Condition A on the RTX 4060 with seed 42 and inspect the result
3. Check the gap between overall accuracy and balanced accuracy — that gap is the
   imbalance problem made visible, and belongs in the results chapter
4. If the baseline is plausible (accuracy 0.75–0.85, macro-F1 0.55–0.70), repeat
   for seeds 43 and 44

---

## Day 5 — Friday 25 September 2026

**Phase:** 1 (baseline classifier — Condition A)
**Machine:** local laptop (RTX 4060), Run 2 (replication)

### Objective
Write the classifier module and establish the Condition A baseline across three
seeds. This is the reference point against which every other condition is read.

### Completed

**`src/classifier.py` written.** `build_model()` returns ResNet-50 with
`IMAGENET1K_V2` weights and the final fully-connected layer replaced with a
7-output layer (23.5 M parameters). `train_classifier()` fine-tunes with AdamW,
a cosine schedule and mixed precision, using the torch 2.6 API
(`torch.amp.autocast('cuda')`, `torch.amp.GradScaler('cuda')`) rather than the
deprecated `torch.cuda.amp` form in the Build Manual.

**The load-bearing line** is the checkpoint-selection criterion:
`if val["macro_f1"] > best_f1`. Selecting on accuracy would systematically
choose the checkpoint most confident about NV — precisely the failure mode this
thesis is about.

**Dataloaders built** with `augment=False` on all three partitions, per D4
(Condition A is resizing and ImageNet normalisation only). The path index is
built once and shared. `assert_no_synthetic_leakage` was run before training,
as it will be before every training call.

**Batch-composition measurement.** Of 50 randomly drawn training batches of 32,
only **3 contained all seven classes**; on average two classes were absent from
any given batch. DF is 1.02% of the training partition, so a batch of 32
contains 0.33 DF images in expectation. The model receives feedback on NV in
every batch and on DF in roughly one batch in three. This is the imbalance
problem expressed at the level of the optimiser, and belongs in the results
chapter.

**Three baseline runs completed** (seeds 42, 43, 44), each evaluated once on the
frozen test partition with 1,000-iteration bootstrap intervals. Metrics written
to `results/run2_local/metrics/`.

| Metric | Mean ± SD | Values (42 / 43 / 44) |
|---|---|---|
| Macro-F1 | **0.634 ± 0.038** | 0.662 / 0.591 / 0.648 |
| Melanoma recall | **0.459 ± 0.051** | 0.509 / 0.461 / 0.407 |
| Balanced accuracy | 0.602 ± 0.037 | 0.625 / 0.559 / 0.621 |
| Accuracy | 0.805 ± 0.010 | 0.814 / 0.795 / 0.806 |
| Macro AUROC | 0.931 ± 0.006 | 0.924 / 0.933 / 0.936 |

Per-class recall (mean ± SD across seeds):

| Class | Recall | n |
|---|---|---|
| nv | 0.947 ± 0.010 | 1004 |
| bcc | 0.643 ± 0.081 | 71 |
| vasc | 0.635 ± 0.027 | 21 |
| akiec | 0.532 ± 0.168 | 52 |
| bkl | 0.511 ± 0.066 | 167 |
| df | 0.483 ± 0.126 | 20 |
| mel | 0.459 ± 0.051 | 167 |

All values fall inside the Build Manual's plausible ranges, and accuracy is
below 0.90 — consistent with an absence of leakage.

### Findings

**The imbalance problem is now measured, not cited.** A correctly trained
ResNet-50 reports 80.5% accuracy while missing more than half of all melanoma
(recall 0.459). The 20-point gap between accuracy (0.805) and balanced accuracy
(0.602) holds across all three seeds. This is the empirical justification for
every metric choice in the study.

**Accuracy is insensitive to what matters.** Across 26 epochs of seed 42,
validation accuracy moved only between 0.78 and 0.82 while validation macro-F1
rose from 0.48 to 0.65. The model was improving substantially and accuracy
barely registered it. Had checkpoints been selected on accuracy, seed 42 would
have selected epoch 6 (accuracy 0.8166, macro-F1 0.6227) instead of epoch 18
(macro-F1 0.6469) — a materially worse model.

**Bootstrap interval width scales inversely with support**, exactly as D13
anticipated:

| Class | n | 95% CI width |
|---|---|---|
| nv | 1004 | 0.029 |
| mel | 167 | 0.149 |
| bcc | 71 | 0.236 |
| akiec | 52 | 0.263 |
| vasc | 21 | 0.439 |
| df | 20 | 0.455 |

DF recall of 0.500 carries an interval of [0.273 – 0.727]. Differences below
roughly 45 percentage points on DF cannot be distinguished from sampling
variation. The pre-registered designation of melanoma as primary and DF/VASC as
secondary (D13) is confirmed by measurement.

**A concern for later conditions.** Melanoma recall is the *least* stable of
the headline metrics across seeds (SD 0.051) while accuracy is the most stable
(SD 0.010) — the useful metric is the noisy one. Any improvement claimed for
the generative arms must exceed this spread; roughly 0.56 would be needed for a
confident claim on three seeds. Consider five seeds if time permits (permitted
by `PRE_REGISTRATION.md` §9).

**AUROC and F1 diverge informatively.** Macro AUROC is 0.931 while macro-F1 is
0.634. The model ranks suspicious images correctly but, at the argmax decision,
imbalance pushes it toward NV. The discriminative information is present; the
decision rule is what fails. This supports D15's treatment of AUROC as
supplementary rather than substitutive.

**Early stopping behaved unevenly across seeds**: it triggered at epochs 25, 9
and 20, with best epochs 18, 2 and 13. Seed 43 stopped after ten epochs because
a favourable epoch-2 score began the patience counter early; its macro-F1
(0.591) is the lowest of the three and its AKIEC recall collapsed to 0.346
against 0.673 for seed 42. Recorded as D19.

**Class distinctiveness matters as well as class size.** VASC has 21 test
images yet the *second most stable* recall (SD 0.027), because vascular lesions
are visually unambiguous. AKIEC has 52 test images and the *least* stable
recall (SD 0.168). Support alone does not predict stability. Worth raising in
the discussion chapter.

### Decisions
**D19 added** — the early-stopping criterion (patience 7, max 30 epochs) is
recorded because it appears in neither the proposal nor the pre-registration.
It was fixed in `config.yaml` and committed in `0e2add07` before any training
run, applied identically to all conditions, seeds and arms, and **was not
tuned**. Tuning would have cost roughly forty GPU-hours and would have optimised
a variable the study holds constant by design. A suboptimal but uniform
criterion shifts every condition's level equally and leaves the differences
intact.

### Problems and resolutions
- An attempt to append D19 by rewriting `DECISION_LOG.md` in PowerShell
  truncated the file to six entries: the `Select-String` pattern failed to match
  the em-dash in the addendum heading. Recovered with `git checkout --
  DECISION_LOG.md`, which restored the committed 18-entry version — a concrete
  demonstration of why the repository exists.
- The retry with `Add-Content` succeeded but corrupted UTF-8 em-dashes
  (`â€"`). Resolved by reading and writing through
  `System.IO.File.ReadAllText/WriteAllText` with an explicit UTF-8 encoding.
  **Rule adopted: edit Markdown through .NET UTF-8 calls, not PowerShell
  redirection.**

### Clarifications recorded
- Validation never updates weights; it only measures. Training loss and
  validation macro-F1 are not comparable quantities — the first reports fit to
  data the model has seen, the second performance on data it has not.
- Bootstrap resamples the *test predictions*, with replacement, at the original
  size. It retrains nothing and changes no image counts. It answers "how much
  would this number move on a different sample of the same size?", which is a
  different question from the quantity ablation (RQ4).
- The study reports all seven classes; macro-F1, a primary outcome, weights them
  equally. Melanoma receives particular attention for a clinical reason (it is
  named in proposal §7.6) and a statistical one (n=167 gives the narrowest
  interval among the minority classes), not because other classes are excluded.

### Open items
- [ ] Condition B (classical augmentation) not yet run
- [ ] Confusion matrices not yet plotted — needed to see what melanoma is
      confused *with*
- [ ] `sampling_pilot.py` not yet run (Kaggle session)
- [ ] Split CSVs not yet uploaded to Kaggle as a private dataset
- [ ] Run 1 (Kaggle, the pre-registered primary execution) not yet begun
- [ ] `gan.py`, `diffusion.py` not yet written
- [ ] Ablation class still to be fixed

### Next session
1. Condition B — classical augmentation, three seeds, `augment=True` on the
   training loader only
2. Plot confusion matrices for A and B; check specifically whether melanoma is
   being absorbed into NV
3. Compare A and B — this is the bar the generative arms must clear (RQ3)

---

## Day 6 — Saturday 26 September 2026

**Phase:** 1–2 (classical augmentation reference; Run 1 environment setup)
**Machine:** local RTX 4060 (Run 2), plus Kaggle setup

### Objective
Establish Condition B, understand *where* the classifier's errors fall, and
begin preparing the pre-registered primary execution on Kaggle.

### Completed

**Condition B — classical augmentation, three seeds.** Identical to Condition A
in every respect except `augment=True` on the training loader. Verified before
running that the transform pipelines differed only where intended:

```
train(aug)  Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation,
            ColorJitter, RandomAffine, ToTensor, Normalize
val         Resize, ToTensor, Normalize
test        Resize, ToTensor, Normalize
```

**Results, A vs B (mean ± SD over seeds 42/43/44):**

| Metric | A baseline | B classical | Diff | A's SD | Beyond noise? |
|---|---|---|---|---|---|
| Macro-F1 | 0.6337 ± 0.0377 | 0.6659 ± 0.0179 | +0.032 | 0.038 | No |
| Melanoma recall | 0.4591 ± 0.0509 | 0.4990 ± 0.0714 | +0.040 | 0.051 | No |
| Balanced accuracy | 0.6015 ± 0.0370 | 0.6553 ± 0.0355 | +0.054 | 0.037 | Marginal |
| Accuracy | 0.8051 ± 0.0097 | 0.8096 ± 0.0139 | +0.004 | 0.010 | No |
| **Macro AUROC** | 0.9308 ± 0.0064 | **0.9502 ± 0.0017** | **+0.019** | 0.006 | **Yes** |

Per-class recall (A → B): vasc +0.111, akiec +0.103, bcc +0.061, df +0.050,
mel +0.040, bkl +0.030, **nv −0.018**. The pattern is consistent — minority
classes gain, the majority class loses slightly. Augmentation shifted the
model's attention away from NV.

**Only AUROC clears the noise floor.** Its improvement (+0.019) is eleven times
its standard deviation (0.0017), and B's spread is *smaller* than A's, so
augmentation also stabilised results. Every other gain is real in direction but
within seed-to-seed variance at three seeds. Melanoma recall moved
inconsistently: −0.090 on seed 42, +0.096 on seed 43, +0.114 on seed 44.

**Overfitting was visibly suppressed.** Condition A drove training loss to
0.0014; Condition B stopped at 0.098. Training also ran longer where it
mattered — seed 43 stopped at epoch 9 under A but ran to epoch 27 under B,
and its macro-F1 rose from 0.591 to 0.686.

**Figure 7 — confusion matrices for A and B**, row-normalised, averaged over
three seeds. This produced the most clinically significant finding so far.

| Melanoma predicted as | A | B |
|---|---|---|
| mel (correct) | 45.9% | 49.9% |
| **nv** | **41.3%** | **33.3%** |
| bkl | 5.4% | 9.0% |
| akiec | 4.0% | 6.0% |
| bcc | 1.6% | 0.8% |
| df | 1.6% | 0.8% |
| vasc | 0.2% | 0.2% |

**Under the baseline, 41.3% of melanoma images were classified as melanocytic
nevi** — the single most consequential error available in this task. Melanoma's
errors are not scattered: of 54% misclassified, 41 points fall into NV alone.
Classical augmentation reduced this to 33.3%, roughly thirteen additional
melanoma across the 167 test images withheld from the benign category.

**But the recovery is only partial.** Of the 8.0 points that left NV, only 4.0
became correct classifications; the remainder moved into BKL (+3.6) and AKIEC
(+2.0). Augmentation weakened the majority-class bias without fully supplying
the discriminative information needed. Clinically this is still an improvement —
BKL and AKIEC both prompt further review, whereas NV does not — but it frames
the question for the generative arms: can synthetic images supply the missing
information, or will they too merely shift the bias?

**NV behaves as an absorbing class.** Every class leaks into it: mel 41%,
bkl 31%, bcc 19% under the baseline. When uncertain, the model answers NV,
because at 67% of training data that is statistically safe. Augmentation
shrank this (NV recall 0.947 → 0.929) to the benefit of every minority class.

**DF fails differently from melanoma.** DF's errors scatter across AKIEC (17%),
BKL (12%) and MEL (12%), with only 5–7% reaching NV. Melanoma's problem is
majority-class bias; DF's is that the model has no reliable notion of what DF
looks like, from 71 training images. This predicts that synthetic DF images will
help only if they carry genuine morphology rather than resampled variation —
directly relevant to interpreting RQ2 alongside RQ1.

**AKIEC's precision collapse explained.** Under B, AKIEC recall rose
0.532 → 0.635 but precision fell (0.556 → 0.391 on seed 42). The confusion
matrix shows why: bkl→akiec rose 2%→10%, bcc→akiec 8%→11%, mel→akiec 4%→6%.
The model is simply predicting AKIEC more often, not recognising it better.

**Run 1 (Kaggle) setup begun.**
- Split CSVs uploaded as a **private** Kaggle dataset
  `ham10000-splits-1229692` (700 kB, 3 files). Verified on Kaggle: test.csv
  shows 1,121 unique lesions and 1,502 unique images, matching local exactly
- **Images uploaded as a private dataset rather than using the public mirror.**
  The community mirror is titled "Skin Cancer MNIST: HAM10000" and ships
  `hmnist_*.csv` files containing 28×28 downsampled copies. MNIST is on the
  University's rejected-dataset list, and a 28×28 file read by mistake would
  invalidate everything downstream. Uploading 2.58 GB removes the risk
  entirely, and keeps the directory layout identical to local
- Zip verified before upload: 10,015 entries, flat structure, first entry
  `ISIC_0024306.jpg`. Source folder confirmed still at 10,015 files afterwards
- Notebook `run1-cloud-baseline` created, accelerator set to GPU T4 ×2

**Run 1 environment recorded:**

| | Run 1 (Kaggle) | Run 2 (local) |
|---|---|---|
| GPU | **Tesla T4, 15,360 MiB** | RTX 4060, 8,590 MiB |
| Driver | 580.159.04 | 566.24 |
| PyTorch | 2.10.0+cu128 | 2.6.0+cu124 |

Only P100 or T4 ×2 were offered; T4 was selected. **Both generative arms of
Run 1 must run on Tesla T4** — `BudgetTracker` raises `DeviceChanged` otherwise.
The differing PyTorch versions are one more reason the two executions are
reported separately and never pooled.

### Clarifications recorded
- Six models were trained in total so far: three for Condition A and three for
  Condition B, each from a fresh `build_model()`. B did not continue from A;
  that would have carried A's training into B and invalidated the comparison.
- Classical augmentation creates **no new files**. The same 6,981 images are
  transformed differently on each epoch, so the model rarely sees an identical
  image twice. This is why it suppresses memorisation but does not alter class
  balance — DF is still 71 images. The generative arms differ precisely here:
  they add files.
- The split is **never regenerated**, on Kaggle or anywhere. Both executions
  read the same committed CSVs. Regenerating risks a different partition
  through library-version differences, and with 20 DF test images a single
  reallocated lesion moves recall by five points.

### Open items
- [ ] Kaggle datasets not yet attached to the notebook (attach was slow)
- [ ] `sampling_pilot.py` not yet run
- [ ] Run 1 Conditions A and B not yet executed
- [ ] `gan.py`, `diffusion.py` not yet written
- [ ] Ablation class still to be fixed

### Next session
1. Attach both datasets to the Kaggle notebook and confirm file counts
2. Clone the repository into the Kaggle session so Run 1 uses identical code
3. Run `sampling_pilot.py` on the T4 to obtain seconds-per-image, and settle the
   Run 1 schedule against measurement rather than estimate
4. Begin Run 1 Condition A

---

## Day 7 — Sunday 27 September 2026

**Phase:** Run 1 environment setup (Kaggle)
**Machine:** Kaggle, CPU session only — GPU quota deliberately preserved

### Objective
Get the pre-registered primary execution environment working and verified, on
CPU, so that no GPU quota is spent on setup.

### Completed

**Images uploaded as a private Kaggle dataset** rather than using the public
community mirror. `ham10000-images-1229692`, 2.58 GB, 10,015 JPEGs, private.
Kaggle extracted the archive automatically. Rationale recorded on Day 6: the
mirror ships `hmnist_*.csv` files containing 28×28 downsampled copies and is
titled "Skin Cancer MNIST", and MNIST is on the University's rejected-dataset
list. Uploading removes the risk rather than managing it, and keeps the
directory layout identical to local.

**Both datasets attached and verified.** Kaggle's current attach mechanism
nests inputs under an extra username level, so the paths are not those in the
Build Manual:

```
IMG  = /kaggle/input/datasets/ashwanisingh997/ham10000-images-1229692
SPL  = /kaggle/input/datasets/ashwanisingh997/ham10000-splits-1229692
CODE = /kaggle/input/datasets/ashwanisingh997/thesis-code-1229692
```

`build_image_index()` resolved all 10,015 images despite the different layout,
because it globs recursively — the fix made on Day 1 for the `part_1`/`part_2`
problem also absorbed this one.

**Splits verified identical to local**, read from the Kaggle copy:

| | train | val | test |
|---|---|---|---|
| Images | 6,981 | 1,532 | 1,502 |
| akiec | 222 | 53 | 52 |
| bcc | 361 | 82 | 71 |
| bkl | 772 | 160 | 167 |
| df | 71 | 24 | 20 |
| mel | 773 | 173 | 167 |
| nv | 4,683 | 1,018 | 1,004 |
| vasc | 99 | 22 | 21 |

All three leakage assertions pass. Every count matches Run 2 exactly, which is
what makes the two executions comparable.

**Code transferred as a private dataset, not by git clone.** The repository is
private, so `git clone` inside a Kaggle cell prompted for a GitHub username and
hung. Entering credentials in a notebook was rejected: GitHub no longer accepts
password authentication, and any token typed into a cell would be saved with
the notebook. Instead `src/` and `config/` were uploaded as
`thesis-code-1229692` (10 files, ~64 kB, private).

Kaggle flattens uploads, so the files arrived without the `src/` directory.
Rather than change the import statements — which would have made Run 1's code
differ from Run 2's — the package layout is reconstructed in `/kaggle/working`
at the start of the session:

```python
for f in os.listdir(CODE):
    dst = f"{WORK}/config/{f}" if f.endswith(".yaml") else f"{WORK}/src/{f}"
    shutil.copy(f"{CODE}/{f}", dst)
sys.path.insert(0, WORK)
```

`from src.data import ...` then works unchanged. **Run 1 therefore executes
byte-identical source to Run 2.**

**Code provenance:** the uploaded code corresponds to commit `3853a1a`. Any
later change to `src/` requires a new version of the Kaggle dataset, and the
commit hash must be recorded again.

**Environment confirmed:**

| | Run 1 (Kaggle) | Run 2 (local) |
|---|---|---|
| GPU | Tesla T4, 15,360 MiB | RTX 4060, 8,590 MiB |
| Driver | 580.159.04 | 566.24 |
| PyTorch | 2.10.0+cu128 (GPU session) | 2.6.0+cu124 |
| Python | 3.12 | 3.11.9 |

Only T4 ×2 and TPU were offered as accelerators; P100 was unavailable. **Both
generative arms of Run 1 must run on Tesla T4**, which `BudgetTracker` enforces
by raising `DeviceChanged` on any resumed segment landing elsewhere.

`init_run("run1_cloud")` succeeded and recorded segment 1. It correctly logged
`cpu`, since the session had no accelerator — the registry records what is
actually there rather than what is intended.

### Decisions
No new decision numbers. The code-transfer mechanism is an implementation
detail, not a research choice: it changes how source reaches the execution
environment, not what the source is.

### Problems and resolutions
- `git clone` of the private repository hung at a username prompt inside a
  Kaggle cell. Resolved by uploading the code as a private Kaggle dataset.
  Credentials were deliberately not entered into a notebook.
- Kaggle's upload flattened the `src/` package. Resolved by reconstructing the
  directory layout in `/kaggle/working` rather than editing imports, so both
  executions run identical code.
- An early cell calling `torch.cuda.get_device_name(0)` raised
  `AssertionError: Torch not compiled with CUDA enabled` — expected, since the
  session was CPU-only. Kaggle ships `2.10.0+cpu` on CPU sessions and
  `2.10.0+cu128` on GPU sessions.

### Note on `/kaggle/working`
Its contents do not survive a session unless the notebook version is saved.
Anything Run 1 produces — metrics, checkpoints, budget state — must be saved
as a notebook version or downloaded before the session ends. This matters
directly for the resumable `BudgetTracker`: its state file lives in the working
directory, and losing it would reset the budget clock.

### Open items
- [ ] `sampling_pilot.py` not yet run — still the missing measurement behind
      the Run 1 schedule
- [ ] Run 1 Conditions A and B not yet executed
- [ ] `gan.py`, `diffusion.py` not yet written
- [ ] Ablation class still to be fixed
- [ ] Output-persistence strategy for Run 1 to be settled before long runs

### Next session
1. Start a **GPU** session and re-run the setup cells (paths, code copy,
   `init_run`) — this time the registry records Tesla T4
2. Run `sampling_pilot.py` on the T4 for seconds-per-image, and fix the Run 1
   schedule against measurement rather than estimate
3. Begin Run 1 Condition A

---

## Day 8 — Monday 28 September 2026

**Phase:** 1–2 complete on both executions
**Machine:** Kaggle, Tesla T4 (Run 1, the pre-registered primary execution)

### Objective
Run the sampling pilot required by D16, then execute Conditions A and B on the
primary environment.

### Completed

**Sampling pilot — the measurement the Run 1 schedule depended on.**
Tesla T4, SD 1.5, fp16, 512 px, batch 8, guidance 7.5, 20 timed images after
warm-up:

| Scheduler | Steps | s/image | Sampling total (15,000) |
|---|---|---|---|
| **PNDM (default)** | **30** | **4.82** | **20.1 GPU-h** |
| DPM-Solver++ | 25 | 4.40 | 18.3 GPU-h |
| DPM-Solver++ | 20 | 3.64 | 15.2 GPU-h |

Peak VRAM 6.47 GB. The saving is sub-proportional — a third fewer steps buys
only a quarter less time — because VAE decoding and pipeline overhead are fixed
per image.

**Dataloader throughput measured, and it changed the whole budget.**

| | ms/batch | s/epoch | min/run (~20 epochs) |
|---|---|---|---|
| num_workers=0 | 406.4 | 89.0 | 29.7 |
| num_workers=2 | 210.0 | 46.0 | 15.3 |
| **num_workers=4** | **110.2** | **24.1** | **8.0** |

Pure compute is 56.6 ms/step, i.e. 12.4 s/epoch. **At `num_workers=0`, 86% of
each epoch was spent decoding JPEGs while the GPU idled.** Earlier estimates of
26–39 GPU-hours for the classifier runs had extrapolated from compute alone and
were badly wrong; the true figure is **3.5 GPU-hours for all 26 runs**.

**Consequence: no pre-registered quantity needed changing.** Scheduler stays at
30 steps, the efficiency curve stays at four checkpoints and 500 images per
class, the synthetic target stays at 1,000. Recorded as **D20**. The pilot's
value was in showing that no reduction was required, not in licensing one.

**Run 1 revised budget: ~32.6 GPU-hours**, comfortably inside two quota weeks.
Quota confirmed on the Kaggle settings page: **30 hours per week** (45 minutes
used at the time of checking).

**Run 1 Conditions A and B executed**, three seeds each, on Tesla T4 with
PyTorch 2.10.0+cu128, code from commit `3853a1a`.

| Metric | A (Run 1) | B (Run 1) | Diff |
|---|---|---|---|
| Macro-F1 | 0.6220 ± 0.0170 | 0.6636 ± 0.0150 | +0.042 |
| Melanoma recall | 0.4251 ± 0.0452 | 0.5070 ± 0.1006 | +0.082 |
| Balanced accuracy | 0.5955 ± 0.0157 | 0.6461 ± 0.0302 | +0.051 |
| Accuracy | 0.7994 ± 0.0037 | 0.8105 ± 0.0034 | +0.011 |
| Macro AUROC | 0.9293 ± 0.0107 | 0.9505 ± 0.0024 | +0.021 |

### Findings

**The replication succeeded.** Every Condition A metric overlaps between the
two executions within seed variance, despite different GPU architecture,
PyTorch version and Python version:

| Metric | Run 1 (T4) | Run 2 (4060) |
|---|---|---|
| Macro-F1 | 0.622 ± 0.017 | 0.634 ± 0.038 |
| Melanoma recall | 0.425 ± 0.045 | 0.459 ± 0.051 |
| Balanced accuracy | 0.596 ± 0.016 | 0.602 ± 0.037 |
| Accuracy | 0.799 ± 0.004 | 0.805 ± 0.010 |
| Macro AUROC | 0.929 ± 0.011 | 0.931 ± 0.006 |

`PRE_REGISTRATION.md` §1 anticipated disagreement and committed to reporting it
as a finding about hardware dependence. There is no disagreement to report at
the reference conditions. This is the strongest possible position from which to
read the generative arms later.

**AUROC is the one gain that replicates cleanly.** Run 1 +0.021 (SD 0.0024),
Run 2 +0.019 (SD 0.0017). The same improvement, to within a thousandth, on
different hardware. Classical augmentation reliably improves how the model
*ranks* suspicious images.

**Melanoma recall remains the unstable metric.** Run 1 Condition B gives
+0.082, the largest gain seen — but its SD is 0.1006, with seeds at 0.617,
0.419 and 0.485. The improvement is smaller than its own spread. Across both
executions, classical augmentation has not yet been shown to improve melanoma
recall; the direction is consistently positive and the magnitude is consistently
unresolvable at three seeds.

**One genuine disagreement between executions: BKL.** Run 1 shows −0.054, Run 2
shows +0.030. Both are small, but they point in opposite directions. Worth
revisiting once all four conditions exist.

**VASC gains most in both executions** — Run 1 +0.190, Run 2 +0.111. Vascular
lesions are round and roughly symmetric, so rotation and flipping are
well-matched to the class.

**Noted, not yet interpreted:** Run 1's standard deviations are consistently
smaller than Run 2's (macro-F1 0.017 vs 0.038). Three seeds give an SD whose own
confidence interval spans roughly 0.6× to 3×, so these ranges overlap heavily
and nothing can be concluded. Re-examine once twelve runs per execution exist.

**Noted for D19:** Run 1 Condition B seed 43 completed all 30 epochs without
early stopping, still improving at epoch 27 (val macro-F1 0.6838). Where seed 43
under Condition A stopped prematurely at epoch 9, here the epoch ceiling bound
instead. Both limits are reported as acting unevenly across seeds.

### Operational
- Kaggle's `/kaggle/working` does not persist; **Save Version → Quick Save**
  after every run that produces results. Done at two points today.
- Run 1 outputs downloaded as a zip and committed to the repository alongside
  Run 2's, in `results/run1_cloud/`.
- `num_workers=4` on Kaggle only. Run 2 keeps `num_workers=0` because Windows
  Jupyter hangs with worker processes. This affects speed, not data.

### Open items
- [ ] `gan.py` and `diffusion.py` not yet written — now the critical path
- [ ] Both generative arms must run **in the same quota week on Tesla T4**;
      a week boundary risks a different accelerator allocation
- [ ] Ablation class still to be fixed
- [ ] Run 1 confusion matrices not yet plotted

### Next session
1. Write `src/gan.py` — conditional WGAN-GP, Equation 7.1, InstanceNorm in the
   critic, five-class label space per D14
2. Verify a few steps run locally, within the 1.64 GB measured on the 4060
3. Upload the updated code as a new Kaggle dataset version and record the
   commit hash

---

## Day 9 — Sunday 4 October 2026

**Phase:** 3
**Session:** 10:22 – ~18:30 UTC (8.1 h wall clock, single session)
**Machine:** Kaggle Tesla T4 (15.64 GB, capability 7.5, driver 580.178.04)

### Objective
Train both generative arms under the declared 4.0 GPU-hour budget, in one
session, on one device.

### Completed
Both arms ran to budget exhaustion in a single segment each, GAN first, then
diffusion. Source: `results/run1_cloud/logs/budget_state_run1_cloud_{gan,diffusion}.json`.

| | GAN | Diffusion |
|---|---|---|
| GPU-hours | 4.0013 | 4.001 |
| Optimiser steps | 3,943 | 2,741 |
| Segments | 1 (Tesla T4) | 1 (Tesla T4) |
| Stop reason | `gpu_hours_exhausted` | `gpu_hours_exhausted` |
| Checkpoints | 1.0 / 2.0 / 3.0 / 4.0 h + final | same |

GAN checkpoint writes took 14.6 s in total, 0.10% of the budget (D17).
Loss history is recorded every 100 steps inside each budget-state file.

**Environment** (`results/run1_cloud/run_meta_generators_session.json`):
Python 3.13.15, PyTorch 2.11.0+cu128, code at commit `5dbc225`.
This differs from the Conditions A/B session of 2 October (Python 3.12.13,
PyTorch 2.10.0, commit `3853a1a`): Kaggle updated its image between sessions.
The equal-budget comparison is unaffected (both arms in one session), but
Conditions C/D will run on a different software version from A/B. To be
disclosed.

Outputs downloaded as `run1_generators.zip` and stored outside the repository
at `D:\thesis_checkpoints\run1_cloud\` (1.8 GB).

### Problems and resolutions
<!-- Fill from memory: any errors hit during the session, e.g. torchao/peft. -->

### Budget consumed
GAN 4.0013 GPU-h, diffusion 4.001 GPU-h, both on Tesla T4.
Kaggle quota after session: 09:19 / 30 h.

---

## Day 10 — Monday 5 October 2026

**Phase:** 4
**Session:** new chat session; handover from `00_SESSION_HANDOVER.md`
**Machine:** local RTX 4060 (inspection only; no Kaggle quota used)

### Objective
Inspect both generators before spending ~20 GPU-hours on sampling; secure the
records of Day 9 in the repository.

### Completed
**GAN training grids inspected by eye** (`gan_samples_{1.0h..final}.png`):
no visible mode collapse, rows clearly class-conditioned, images still
changing between 3.0 h and 4.0 h at fixed noise, 4.0 h and final
indistinguishable. Checkerboard texture visible on plain backgrounds.

**Both arms then inspected with one identical procedure**
(`inspect_generators.py`, commit `46bcb11`; results in
`results/run1_cloud/generator_inspection.json`). The first plan — GAN by eye,
diffusion by numbers — was asymmetric and was replaced. 32 samples per class
per arm, same 8-bit 128 px pathway as the classifier, 32 real training images
per class as reference.

| | Real | GAN | Diffusion |
|---|---|---|---|
| Spread ratio to real | 1.0 | 0.76 – 0.87 | 0.54 – 0.76 |
| Near-duplicate pairs | — | none | none |
| Label / noise effect | — | 0.76 | 0.70 |
| Mean RGB (approx.) | 195, 148, 156 | 193, 151, 159 | 166, 117, 145 |

- LoRA adapter verified: 1.59M parameters, max |lora_B| = 0.042, mean pixel
  change vs base SD 1.5 at identical seeds 43–53. Base SD 1.5 produces no
  dermatoscopic images at all; the domain was learned in the 4 GPU-hours.
- Each arm carries its own artefact: GAN checkerboard texture, diffusion
  colour shift (darker, toward magenta). Both are possible classifier shortcuts.
- Diffusion: noise fixes the lesion layout, the prompt changes colour and
  texture (same-seed columns share composition across classes).
- Pixel-space spread is coarse: checkerboard inflates GAN variance, uniform
  background deflates diffusion variance. FID and LPIPS will decide diversity.
- Real DF has the smallest real min-pair distance (0.072), consistent with
  multiple photographs per lesion (Figure 4).
- Local sampling rate: 160 LoRA images in 12.4 min, ~4.65 s/image on the
  RTX 4060 (cf. 4.82 s/image on T4, D20). Useful for planning Run 2.

**Repository records secured** (commit `473a2eb`): both budget-state logs and
the generator-session `run_meta` copied into `results/run1_cloud/`.

**Data protection checked:** `data/raw/` and `data/synthetic/` ignored and never
committed in history; only the three split CSVs are tracked. A gap was found
and closed: `.gitignore` named `results/checkpoints/` while outputs live under
`results/run1_cloud/` and `results/run2_local/`, so synthetic PNGs would have
been committable. Replaced with `results/*/checkpoints/`, `results/*/synthetic/`,
`results/*/samples/`; verified with `git check-ignore` that figures remain tracked.

**Analysis plan verified:** `PRE_REGISTRATION.md` has exactly one commit,
`02d7af7`, 23 Sep 2026 23:48, never amended, made before `gan.py` existed.

### Decisions
- **D21** — headline sampling from `final` checkpoints of both arms; observed
  artefacts reported, not corrected (proposal §7.3).
- **D22** — real-to-real calibration excludes same-lesion images (proposal §7.5,
  "distinct real training images").
- **Considered and rejected:** extending both arms beyond 4.0 GPU-hours because
  the GAN was still changing. Rejected because the budget was declared in
  advance, the decision would be conditioned on observed output from one arm
  only, "changing" is not "improving", and the efficiency curve (D9) already
  answers the question by design.

### Problems and resolutions
- `pipe.enable_vae_slicing()` no longer exists in the installed diffusers;
  replaced by `pipe.vae.enable_slicing()` in the inspection script.
  **The same call exists in `src/diffusion.py` `build_pipeline`**, guarded by
  VRAM < 12 GB: never reached on the T4 (Run 1 safe), but will crash Run 2 on
  the 8 GB RTX 4060. Must be fixed before Run 2; changing `src/` requires a new
  Kaggle code version and recorded hash.

### Budget consumed
None. Inspection on the local GPU from Run 1 weights, outside the budget (D2).

### Open items
- [ ] Upload Run 1 checkpoints to a **private** Kaggle dataset (exclude `*_resume`)
- [ ] Pin the Kaggle notebook environment before C/D; record as a decision
- [ ] Fix `enable_vae_slicing` in `src/diffusion.py` before Run 2
- [ ] D22 sub-decisions before the audit: pooled vs per-class threshold;
      fixed class for the controlled sub-experiment (pre-registration §7, §9)
- [ ] Save the Day 9 Kaggle notebook as `notebooks/04_run1_generators.ipynb`
- [ ] Complete Day 9 "Problems and resolutions" from memory
- [ ] Review and commit the pending change in `notebooks/03_figures.ipynb`
- [ ] Email `PRE_REGISTRATION.md` to the supervisor before C/D results exist
- [ ] Later: FFT figure of checkerboard artefact; move root `logs/` to
      `results/run2_local/logs/`; missing `B_classical_summary.csv` in Run 2

### Next session
Upload the Run 1 checkpoints to a private Kaggle dataset, then start headline
sampling (5 classes × 1,000 per arm) from the final checkpoints.

---

## Day 11 — Tuesday 6 October 2026

**Phase:** 4
**Machine:** Kaggle Tesla T4 (sampling); local RTX 4060 (preparation)

### Completed
- Run 1 checkpoints uploaded as private dataset `run1-checkpoints-1229692`
  from `run1_generators.zip` exactly as downloaded (38 files, 2,035,293,609 bytes).
  Per-file SHA-256 manifest written (`results/run1_cloud/checkpoint_manifest.csv`);
  digest `81280427...672dbf2` matched on Kaggle before every later session.
- New notebook `run1-cloud-sampling`. A two-image smoke test found two bugs
  before the full run:
  1. `resumable_generate` saved to `*.png.tmp`; PIL cannot infer a format from
     `.tmp` (ValueError). Fixed with an explicit format.
  2. `pipe.enable_vae_slicing()` removed in current diffusers; replaced by
     `pipe.vae.enable_slicing()` (path only reached on GPUs under 12 GB, i.e. Run 2).
  Commit `99ebf8d`; Kaggle code dataset v3. Training code unchanged.
- Headline sampling (D7, D21) from final checkpoints, committed run of 7 h 5 m:
  GAN 5,000 images in 34 s (0.007 s/image); diffusion 5,000 in 7.04 h
  (5.069 s/image). All 128x128 RGB, no leftover temporary files.
  Environment identical to the generator session (Python 3.13.15, PyTorch 2.11.0).

### Problems and resolutions (Day 9, completed from the session log)
- `torchao 0.10.0` preinstalled on Kaggle broke `peft`; uninstalled at the start
  of the session and verified with `peft.import_utils.is_torchao_available() == False`
  before training. The same step is used in every later session.

---

## 7 to 26 October 2026

No project work.

---

## Day 12 — Tuesday 27 October 2026

**Phase:** 4
**Machine:** local (preparation); Kaggle Tesla T4 (classifiers)

### Completed
- Proposal result received (15 Oct): pass, 54. Deadline 2 December 2026.
  Every marker point logged in the Proposal Feedback Tracker with its response
  (address / justify / limitation), where it is answered, and status.
- Sampling records committed (`dfafc90`).
- **D23** recorded before any C/D result: per-image test predictions saved;
  A and B re-executed with C and D in one session; macro precision, per-class
  specificity and macro AUPRC added; paired bootstrap and McNemar comparison
  fixed in advance. `src/metrics.py` patched (`scripts/patch_metrics_d23.py`),
  tested locally against scikit-learn; commit `f568fe0`, Kaggle code v4.
- Synthetic headline set stored as private dataset
  `run1-synthetic-headline-1229692` (10,000 files).
- Notebook `run1-cloud-classifiers`: smoke test (1 epoch) passed; full run
  started (A, B, C, D x seeds 42/43/44).

---

## Day 13 — Wednesday 28 October 2026

**Phase:** 4–5

### Completed
- Classifier run complete: 12 runs in 2.19 GPU-h, 12 prediction files (commit `7a36014`).

| Condition | Macro-F1 | Melanoma recall |
|---|---|---|
| A baseline (re-run) | 0.620 ± 0.048 | 0.469 ± 0.021 |
| B classical (re-run) | 0.663 ± 0.019 | 0.445 ± 0.023 |
| C GAN | 0.633 ± 0.013 | 0.493 ± 0.034 |
| D diffusion | 0.636 ± 0.016 | 0.409 ± 0.009 |

- Paired analysis per D23 (`scripts/paired_comparison.py`): no difference
  consistent across all three seeds for any declared pair. D − C melanoma recall
  −0.096 / −0.054 / −0.102, two of three intervals excluding zero.
- Reproducibility: re-executed A/B macro-F1 matches 2 October (0.620 vs 0.622;
  0.663 vs 0.664). Melanoma recall does not: B − A was +0.082 on 2 October and
  −0.024 now, confirming melanoma recall is within seed and environment noise.
- **D24** (audit design, DF as sub-experiment class, what the sub-experiment
  can show) and **D25** (FID computation, extra DF images) recorded before any
  distance or FID was computed.
- `src/memorization.py` and `src/fid.py` written (`6c8939b`, Kaggle code v5).
  LPIPS embedding verified exact on the RTX 4060 (max error 1.79e-07);
  same-lesion exclusion verified (0 same-lesion neighbours).
- Notebook `run1-cloud-audit`: smoke test passed; regenerated DF images matched
  the stored ones exactly (mean absolute difference 0.0 for both arms). Full run started.

### Decisions
- Considered MEL for the sub-experiment first; changed to DF before any audit
  result, for the reasons recorded in D24. Also corrected the rationale: drawing
  more samples does not change a generator, so the rate is expected to be flat.

---

## Day 14 — Thursday 29 October 2026

**Phase:** 5

### Completed
- Audit, FID and DF ablation complete (1.87 GPU-h, commit `fed5867`).
  - tau = 0.1333. Zero near-duplicates in either arm, headline and every ablation
    quantity. Closest synthetic image: GAN 0.1439, diffusion 0.1432 (margin about 0.01).
  - FID pooled: GAN 206.7, diffusion 137.5; diffusion lower in four of five
    classes, equal for VASC.
  - Ten closest pairs per arm inspected by eye: no visible copies; layout similar,
    lesions different.
- Exploratory (not declared; `scripts/audit_hubness.py`): 32.8% of GAN images,
  and 696 of 1,000 GAN DF images, share one nearest DF training image
  (ISIC_0030665) that only 2 real images choose; diffusion's largest hub 7.1%.
  Consistent with partial mode concentration in the GAN's DF class; not copying.
- Analysis scripts moved to `scripts/`. `scripts/verification_checks.py`
  re-runs every check first done by hand: 9/9 PASS, including tau recomputed on
  the RTX 4060 = Kaggle T4 (same nearest neighbour for 99.9% of 1,526 images).
  Commit `2d672b6`.
- **D26** (efficiency curve: three seeds per point, shared noise) recorded
  before any curve image (`0869187`).
- Curve sampling part a started (1.0 h and 2.0 h checkpoints).

---

## Day 15 — Saturday 31 October 2026

**Phase:** 5

### Completed
- Curve part a complete: 7.65 h session, 10,000 images; diffusion 5.473 s/image
  (5.069 in the headline session on the same GPU model). Stored as private dataset
  `run1-curve-a-1229692`; logs committed (`01ed5af`).
- Curve part b started (3.0 h and 4.0 h checkpoints). Kaggle quota reset: 30 h available.

### Budget consumed (sampling and evaluation, outside the training budget, D2)
| Session | GPU-h |
|---|---|
| Headline sampling (6 Oct) | 7.09 |
| Classifiers A–D (27 Oct) | 2.19 |
| Audit + FID + DF extra (28 Oct) | 1.87 |
| Curve part a (29–30 Oct) | 7.65 |

### Open items
- [ ] Curve part b, then 24 classifier runs for the curve
- [ ] Quantity ablation for RQ4
- [ ] `notebooks/04_results_analysis.ipynb` for all Chapter 5 tables and figures
- [ ] Download Kaggle notebooks and logs into the repository; `docs/PROVENANCE.md`
- [ ] Email to Dr. Biswas (Run 2, convergence critique, test choice)
- [ ] Commit or discard the pending change in `notebooks/03_figures.ipynb`

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
