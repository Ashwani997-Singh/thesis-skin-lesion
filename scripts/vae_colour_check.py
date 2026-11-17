"""
scripts/vae_colour_check.py - D29 item 5: where does the diffusion colour shift come from?

Generator inspection (D21) found diffusion images darker than real ones in R
and G by about 25-30 levels (generator_inspection.json). Two explanations:

  (a) the Stable Diffusion VAE itself shifts dermoscopy colours when it
      encodes and decodes an image (a pipeline property), or
  (b) the LoRA-adapted model learned the wrong colours in 4.0 GPU-hours
      (a training property).

Test: pass REAL training images through the SD 1.5 VAE only (encode, decode,
no diffusion, no LoRA) with exactly the preprocessing used in training
(512 px, [-1, 1], src/data.PromptedLesions), then compare mean RGB before and
after at the 128 px the classifier sees. A small round-trip shift means the
VAE is not the cause, so the shift was learned (b).

Diagnostic only: no training, outside the budget (D2), not a reported result.
Run on the laptop GPU from the repository root:

    python scripts/vae_colour_check.py | Tee-Object results/run1_cloud/logs/vae_colour_check_output.txt
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import AUG_CLASSES, build_image_index, generator_source_frame  # noqa: E402
from src.diffusion import BASE  # noqa: E402

IMAGES = Path(r"D:\thesis\data\raw\HAM10000\images")
N_PER_CLASS = 30
OUT = ROOT / "results" / "run1_cloud" / "vae_colour_check.json"
INSPECTION = ROOT / "results" / "run1_cloud" / "generator_inspection.json"

dev = "cuda" if torch.cuda.is_available() else "cpu"
from diffusers import AutoencoderKL  # noqa: E402
vae = AutoencoderKL.from_pretrained(BASE, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()

to512 = T.Compose([T.Resize((512, 512)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
to128 = T.Resize((128, 128))

frame = generator_source_frame(ROOT / "data" / "splits" / "train.csv")
index = build_image_index(IMAGES)
rng = np.random.default_rng(42)

def mean_rgb_128(x):          # x in [-1, 1], shape 3xHxW -> 0-255 mean per channel at 128 px
    x = to128(((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)).float()
    return x.mean(dim=(1, 2)).cpu().numpy()

results, all_diff = {}, []
with torch.no_grad():
    for c in AUG_CLASSES:
        ids = frame[frame.dx == c].image_id.values
        pick = rng.choice(ids, size=min(N_PER_CLASS, len(ids)), replace=False)
        before, after, absdiff = [], [], []
        for i in pick:
            x = to512(Image.open(index[i]).convert("RGB")).unsqueeze(0).to(dev)
            z = vae.encode(x).latent_dist.mean
            y = vae.decode(z).sample
            before.append(mean_rgb_128(x[0])); after.append(mean_rgb_128(y[0]))
            absdiff.append(float(((x - y).abs() * 127.5).mean()))
        b, a = np.mean(before, 0), np.mean(after, 0)
        results[c] = {"n": len(pick), "real_mean_rgb": b.round(1).tolist(),
                      "after_vae_mean_rgb": a.round(1).tolist(),
                      "shift_rgb": (a - b).round(1).tolist(),
                      "mean_abs_pixel_diff_0_255": round(float(np.mean(absdiff)), 2)}
        all_diff.append(a - b)
        print(f"{c:6s} real {b.round(0)}  after VAE {a.round(0)}  shift {(a - b).round(1)}  "
              f"|pixel diff| {np.mean(absdiff):.2f}")

vae_shift = np.mean(all_diff, 0)
insp = json.loads(INSPECTION.read_text())
diff_shift = np.mean([np.array(insp["diffusion"]["per_class"][c]["mean_rgb"]) -
                      np.array(insp["real"][c]["mean_rgb"]) for c in AUG_CLASSES], 0)
print(f"\nmean shift from the VAE round trip      (R, G, B): {vae_shift.round(1)}")
print(f"mean shift of diffusion samples (D21)   (R, G, B): {diff_shift.round(1)}")
share = float(np.abs(vae_shift).sum() / np.abs(diff_shift).sum())
print(f"VAE round trip explains about {share:.0%} of the colour shift (by summed absolute shift)")
verdict = ("VAE is NOT the main cause: the shift was learned by the adapted model"
           if share < 0.25 else "VAE contributes substantially to the shift")
print(verdict)

OUT.write_text(json.dumps({"n_per_class": N_PER_CLASS, "device": dev, "per_class": results,
                           "vae_shift_rgb": vae_shift.round(2).tolist(),
                           "diffusion_shift_rgb_D21": diff_shift.round(2).tolist(),
                           "share_explained_by_vae": round(share, 3), "verdict": verdict}, indent=2))
print(f"saved {OUT}")
