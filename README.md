# A Controlled Equal-Budget Comparison of GAN and Diffusion Data Augmentation for Imbalanced Skin-Lesion Classification with a Memorization Audit

MSc dissertation · Machine Learning and Artificial Intelligence
Liverpool John Moores University, delivered with upGrad

**Ashwani Singh** · Student ID 1229692
Supervisor: Dr. Sitanath Biswas

---

## What this study does

HAM10000 is dominated by benign melanocytic nevi, so a classifier trained on it
can report high overall accuracy while failing on the rare malignant categories
that matter clinically. Generative augmentation is a common response, but two
questions remain open: whether adversarial or diffusion models deliver more
benefit **when the training budget is held constant**, and whether the synthetic
images they produce **reproduce identifiable training data**.

This study measures both, on the same benchmark, under one protocol.

### The two contributions

1. **Equal compute budget.** Published comparisons allow each model to train to
   its own convergence, so compute is confounded with method. Here the budget is
   declared in advance (4.0 GPU-hours per arm) and applied identically to both.
2. **Memorization audit paired with downstream benefit.** Augmentation studies
   rarely test whether the generator copies its training data; memorization
   studies rarely connect copying to classification gain. Both are measured here.

### Experimental design

Four training sets, one classifier, one frozen test partition.

| Condition | Training data |
|---|---|
| A Baseline | Original data; resizing and ImageNet normalisation only |
| B Classical | Original + geometric and photometric augmentation |
| C GAN | Original + conditional WGAN-GP synthetic images |
| D Diffusion | Original + LoRA latent-diffusion synthetic images |

Classifier held constant throughout: ResNet-50, ImageNet-pretrained, selected on
validation macro-F1 — never on accuracy.

The protocol is executed twice: **Run 1** on a cloud GPU (pre-registered as
primary, matching the approved proposal) and **Run 2** on a local RTX 4060
(replication). Results are reported separately and never pooled.

---

## Dataset

HAM10000 (Tschandl, Rosendahl and Kittler, 2018), obtained from the Harvard
Dataverse under its non-commercial research licence:

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

10,015 dermatoscopic images across seven diagnostic categories — but only
**7,470 unique lesions**, because many lesions were photographed more than once.

| Class | Images | Lesions | Share |
|---|---|---|---|
| nv | 6,705 | 5,403 | 66.9% |
| mel | 1,113 | 614 | 11.1% |
| bkl | 1,099 | 727 | 11.0% |
| bcc | 514 | 327 | 5.1% |
| akiec | 327 | 228 | 3.3% |
| vasc | 142 | 98 | 1.4% |
| df | 115 | 73 | 1.1% |

**The images themselves are not in this repository.** The licence does not permit
redistribution, and `data/raw/` is excluded by `.gitignore`. Only the split
definitions are version-controlled.

---

## Repository structure

```
config/config.yaml        Every hyper-parameter, in one place
data/splits/              train.csv, val.csv, test.csv - the frozen partition
notebooks/                01_eda.ipynb
results/figures/          All figures, 300 DPI
src/
  utils.py                Seed fixing across Python, NumPy and PyTorch
  data.py                 Path indexing, datasets, matched generator loaders
  split.py                Lesion-level stratified splitting
  budget.py               Resumable, device-pinned GPU-hour enforcement
  checkpointing.py        Crash-safe state; deterministic resumable sampling
  metrics.py              Equations 7.6-7.10, macro AUROC, bootstrap intervals
  run_registry.py         Keeps the two executions separate and auditable
gpu_probe.py              Hardware characterisation
sampling_pilot.py         Measures seconds-per-image before scheduling
PRE_REGISTRATION.md       Analysis plan, fixed before any generative training
DECISION_LOG.md           D1-D18, each checked against the proposal wording
WORK_LOG.md               Dated record of every working session
```

---

## The frozen split

Partitioning is at **lesion level**, not image level. HAM10000 contains multiple
photographs of the same physical lesion; splitting by image would place two views
of one lesion in different partitions, which is data leakage in the sense of
Kaufman et al. (2012).

Duplication is measured and is **not uniform across classes**:

| Class | Lesions with >1 photograph |
|---|---|
| mel | **62.5%** |
| df | 46.6% |
| bcc | 46.5% |
| bkl | 39.5% |
| vasc | 34.7% |
| akiec | 33.8% |
| nv | **18.3%** |

Melanoma is the most heavily duplicated class — clinicians photographed
suspicious lesions more thoroughly — so an image-level split would leak most
severely on precisely the class declared as the primary outcome.

The split was committed to version control **before any generative model
existed**:

```
commit 0e2add07fc53bd9a62760b187c4301f1165917fe
Wed 23 Sep 2026 23:33:52 +0400
```

`data/splits/test.csv` is not regenerated, reshuffled, or allowed to receive a
synthetic image under any condition.

---

## Reproducing the environment

```bash
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Verified on: PyTorch 2.6.0+cu124, Python 3.11.9, Windows 11,
NVIDIA RTX 4060 Laptop GPU (8 GB).

To reproduce the split from the metadata:

```python
from src.split import lesion_level_split, check_split
splits = lesion_level_split(df, seed=42)
check_split(splits, df)
```

Seed 42 reproduces the committed partition exactly.

---

## Hardware characterisation

`gpu_probe.py` measures VRAM, throughput and thermal behaviour before training.
On the RTX 4060, over a five-minute sustained WGAN-GP load:

- SM clock held at 2625 MHz with **no measurable drift**
- Temperature stable at 61–65 °C
- WGAN-GP at 128 px, batch 64: peak **1.64 GB**, 1069 ms per generator step

Absence of throttling matters here: it means four GPU-hours represents the same
quantity of computation at the end of a run as at the start, which the
equal-budget control depends on.

---

## Status

| Phase | Status |
|---|---|
| 1 — Setup, dataset, lesion-level split, baseline | In progress |
| 2 — Classical augmentation reference | Not started |
| 3 — Generative model training | Not started |
| 4 — Synthesis and evaluation | Not started |
| 5 — Memorization audit and ablation | Not started |
| 6 — Analysis and writing | Not started |

---

## Academic integrity

All prose in the dissertation is written by the author. Every citation is
verified against the original paper. AI assistance was used for structuring,
debugging and review, and is declared in the submitted AI-use declaration.

This is a **benchmark study on a public research dataset**. It makes no claim
regarding diagnostic performance in clinical practice, and no prospective or
clinical evaluation is undertaken.

---

## Key references

Tschandl, Rosendahl and Kittler (2018) — HAM10000 ·
Goodfellow et al. (2014) — GANs ·
Gulrajani et al. (2017) — WGAN-GP ·
Ho, Jain and Abbeel (2020) — DDPM ·
Rombach et al. (2022) — latent diffusion ·
Hu et al. (2021) — LoRA ·
Lucic et al. (2018) — equal-budget comparison ·
Akbar, Wang and Eklund (2023) — memorization in medical generative models ·
Zhang et al. (2018) — LPIPS ·
Kaufman et al. (2012) — data leakage

Full reference list in the submitted research proposal.
