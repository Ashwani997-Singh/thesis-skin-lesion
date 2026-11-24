# Proposal feedback (26 points) mapped to the dissertation draft, checked 24 Nov 2026 (doc rev 102)

Source files: Proposal Feedback Tracker.pdf (27 Oct, all "Not started", ORIGINAL version, never updated) and Supervisor Briefing (2 Nov, pre-D28/D29). Both are historical; Appendix B is built from this map, not from those PDFs.
Codes: F1-F10 = First Marker (10), S1-S16 = Second Marker (16). Tracker PDF layout puts F7-F10 under the Second Marker heading by page break; the count text ("Ten points" / "Sixteen points") confirms F = First, S = Second.

| # | Point | Where answered in draft | State |
|---|---|---|---|
| F1 | Premise strong, execution flawed | via F5 answers: Ch3 3.7/3.16/3.17, Ch5 5.8, 5.10 | Done |
| F2 | Testable hypotheses | Ch1 1.4 H0/H1 for RQ1, RQ3; Ch3 3.2 | Done |
| F3 | Why classical aug fails on dermatoscopy | Ch2 2.2 ABCD rule, Nachbar et al. 1994 | Done |
| F4 | Why diffusion may memorise more | Ch2 2.6 Gu et al. 2025, Kadkhodaie et al. 2024, Akbar 2025; Ch5 5.15 | Done |
| F5 | Fixed budget, no HP search, unconverged | Ch3 3.8/3.16/3.17 (D28 policy), Ch5 5.4 FID, 5.8 curve + curve FID, 5.10 tuning | Done |
| F6 | Contribution if zero improvement | Ch1 significance; Ch6 6.2.3/6.3.2 | Ch6 PENDING |
| F7 | 16 GB GPU will OOM | Ch3 3.3 (no OOM, 6.47 GB peak), 3.9.2; Ch5 5.2 | Done |
| F8 | Schedule unrealistic | Ch5 5.2 smoke tests and two faults | Partial: no planned-vs-actual timeline yet (Appendix or 5.2 table) |
| F9 | Critiques of medical synthesis | Ch2 2.7 Chen et al. 2021, Kazerouni 2023, Kynkaanniemi 2023 | Done (Ghorbani DermGAN not used, optional) |
| F10 | Tone procedural | all chapters, limitations in each method section; 5.16 | Done (judgement) |
| S1 | Define fair comparison and audit success | Ch3 3.7 + Table 3.7.2; 3.15 | Done |
| S2 | Distinguish from GAN-vs-diffusion work | Ch2 2.4, 2.8 Table 2.8.1 (Muller-Franzes 2023) | Done |
| S3 | Same images/compute/iterations/ratio | Ch3 3.7 Table 3.7.2 (3,943 vs 2,741) | Done |
| S4 | Justify architectures; instances not families | Ch1 1.5 scope, Ch3 3.9, Ch5 5.11 | Done; repeat in Ch6 |
| S5 | Operational audit; membership inference | Ch3 3.15; Ch5 5.16; Ch6 6.5 | Done; MI in 6.5 |
| S6 | Artefacts, minority bias | Ch5 5.3 artefacts, 5.5 per-class precision (MEL 0.536, NV->MEL 113), 5.9 DF precision | Done |
| S7 | Research-framework figure | Ch3 Figure 3.1.1 PLACEHOLDER only | PENDING: figure must be drawn |
| S8 | Statistically significant aim | Ch3 3.13 D23 bootstrap + McNemar; Ch1 aims | Done |
| S9 | Objectives incl. statistics, per-class | Ch1 objective 5 | Done |
| S10 | GAN vs diffusion trade-offs + memorisation | Ch2 2.3-2.6, trade-off table | Done |
| S11 | Compare recent skin-lesion GAN/diffusion | Ch2 2.3/2.4 Baur, Bissoto, Qin, Su, Akrout, Farooq, Pritam; Table 2.8.1 | Done |
| S12 | Macro-precision, per-class specificity, AUPRC | Ch3 3.12 defines all; Ch5 Table 5.5.1 has macro precision, AUPRC | GAP: per-class specificity defined (Ch3, objective 5) but NOT reported in Ch5 |
| S13 | Ratio ablations | Ch5 5.8, 5.9 | Done |
| S14 | Real-only and classical baselines | Conditions A, B throughout | Done |
| S15 | Outdated refs | Hu et al. 2022 (ICLR), Akbar 2025 published | Done; check rest in reference pass |
| S16 | Presentation weak | figures alongside tables throughout | Done (judgement) |

## Open actions from the map
1. S12: compute per-class specificity from the saved predictions (*_predictions.npz) and add a column/table in 5.5 (or Appendix) - cheap, no GPU.
2. S7: draw Figure 3.1.1 research framework (preprocessing -> split -> equal-budget generators -> sampling + quality validation (FID, inspection) -> classification A-D -> audit -> statistics).
3. F8: planned (proposal Gantt) vs actual timeline table from WORK_LOG, Appendix or 5.2.
4. F6, S4, S5: close in Chapter 6.

## Outdated statements in the 2 Nov briefing (do NOT copy into the thesis)
- "published defaults" -> D28: "standard, untuned settings".
- "Diffusion images were more realistic" -> FID = closer to real images in Inception features.
- RQ4 "diffusion raised macro-F1 gradually 0.620 to 0.668" -> report as in 5.9 (DF recall peak 0.650 at 500, then 0.533).
- Run 2 "report A and B, secondary" -> D29: Run 2 not reported.
- 7.09 GPU-h headline sampling -> 7.05 (sampling generation) / 7.04 h diffusion.
- "Eight Kaggle sessions, ~45 GPU-h" -> final count 10 sessions (2 Oct-16 Nov).
- "all sessions completed" memory 5.0/1.6 GB -> peak 6.47 GB measured in sampling pilot.
