"""
inspect_generators.py - Phase 4: identical inspection of BOTH Run 1 generators.

Supersedes inspect_gan_samples.py and inspect_lora_samples.py. The GAN was
first judged by eye and the diffusion arm was going to be judged by numbers;
an asymmetric check. Here ONE function evaluates both arms, with:

  - the same number of samples per class (N)
  - the same image pathway: generated -> 8-bit 128 px image -> tensor,
    exactly the form the classifier will receive
  - the same real reference images
  - the same statistics:
      spread    per-pixel std across N samples of a class, averaged
      ratio     spread / spread of N real images of that class (~1 healthy, ~0 collapse)
      min_pr    smallest distance between two samples (~0 = near-identical pair)
      label/noise  same noise, different class, vs same class, different noise
                   (~0 = class label ignored)
      mean_rgb  average colour, compared with the real images

The noise differs in kind between arms (GAN: a 128-d vector; diffusion: a
seeded 4x64x64 latent), so "same noise" is only meaningful WITHIN an arm.

Diffusion-only extra: an adapter-loading check against base SD 1.5. This is
a software check (did the weights load?), not an evaluation metric; the GAN
has no base model, so it has no equivalent.

Inspection only: outside the budget (D2), not a reported result, run on the
local RTX 4060 from Run 1 weights. Outside src/, so no new Kaggle version.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\thesis")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image

from src.data import (AUG_CLASSES, CLASS_PROMPTS, HAM10000, build_image_index,
                      generator_source_frame)

CKPT = Path(r"D:\thesis_checkpoints\run1_cloud\checkpoints")
FIG = Path(r"D:\thesis_checkpoints\run1_cloud\figures")
OUT_JSON = ROOT / "results" / "run1_cloud" / "generator_inspection.json"

N = 32            # samples per class, BOTH arms and the real reference
SEED0 = 2026      # noise seeds SEED0+i, shared across classes within an arm
SIZE = 128        # what the classifier sees
GRID_PER_CLASS = 4

cfg = yaml.safe_load(open("config/config.yaml"))
device = "cuda"
to_tensor = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])


# --- shared evaluation code: used identically for both arms ------------------

def stats(x):
    flat = x.flatten(1).float()
    d = torch.cdist(flat, flat) / flat.shape[1] ** 0.5
    off = d[~torch.eye(len(x), dtype=torch.bool)]
    return {
        "spread": round(x.std(dim=0).mean().item(), 4),
        "min_pair": round(off.min().item(), 4),
        "mean_pair": round(off.mean().item(), 4),
        "mean_rgb": ((x.mean(dim=(0, 2, 3)) + 1) * 127.5).round().int().tolist(),
        "has_nan": bool(torch.isnan(x).any()),
    }


def evaluate(fake, real_stats):
    """fake: {class: tensor [N,3,128,128]}, sample i of every class from noise i."""
    per_class = {}
    for c in AUG_CLASSES:
        s = stats(fake[c])
        s["ratio_to_real"] = round(s["spread"] / real_stats[c]["spread"], 3)
        per_class[c] = s
    label_eff = []
    for i, a in enumerate(AUG_CLASSES):
        for b in AUG_CLASSES[i + 1:]:
            diff = (fake[a] - fake[b]).flatten(1)
            label_eff.append(diff.pow(2).mean(dim=1).sqrt().mean().item())
    label = float(np.mean(label_eff))
    noise = float(np.mean([per_class[c]["mean_pair"] for c in AUG_CLASSES]))
    return {"per_class": per_class, "label_effect": round(label, 4),
            "noise_effect": round(noise, 4),
            "label_over_noise": round(label / noise, 3)}


def print_table(title, r):
    print(f"\n{title}")
    print(f"{'class':6} {'spread':>7} {'ratio':>6} {'min_pr':>7} {'mean_rgb':>16} nan")
    for c, s in r["per_class"].items():
        print(f"{c:6} {s['spread']:7.4f} {s['ratio_to_real']:6.3f} "
              f"{s['min_pair']:7.4f} {str(s['mean_rgb']):>16} {s['has_nan']}")
    print(f"label_effect={r['label_effect']:.4f}  noise_effect={r['noise_effect']:.4f}"
          f"  label/noise={r['label_over_noise']:.3f}")


def save_grid(pils, path):
    grid = Image.new("RGB", (SIZE * GRID_PER_CLASS, SIZE * len(AUG_CLASSES)), "white")
    for r, c in enumerate(AUG_CLASSES):
        for k in range(GRID_PER_CLASS):
            grid.paste(pils[c][k], (k * SIZE, r * SIZE))
    grid.save(path)
    print(f"Saved {path}")


def as_tensor(pils):
    return {c: torch.stack([to_tensor(p) for p in pils[c]]) for c in AUG_CLASSES}


# --- real reference ----------------------------------------------------------

print("Loading real reference images ...")
frame = generator_source_frame(Path(cfg["paths"]["splits"]) / "train.csv")
index = build_image_index(cfg["paths"]["local_images"])
real_stats = {}
for c in AUG_CLASSES:
    sub = frame[frame.dx == c].sample(n=N, random_state=0)
    ds = HAM10000(frame=sub, image_root=None, size=SIZE, class_space="augmented",
                  normalise="tanh", index=index)
    real_stats[c] = stats(torch.stack([ds[i][0] for i in range(len(ds))]))
results = {"N": N, "real": real_stats}


# --- arm 1: GAN --------------------------------------------------------------

from src.gan import Generator

latent = cfg["gan"]["latent_dim"]
G = Generator(latent_dim=latent)
G.load_state_dict(torch.load(CKPT / "gan_final.pt", map_location="cpu",
                             weights_only=False)["models"]["G"])
G.to(device).eval()

z = torch.randn(N, latent, generator=torch.Generator().manual_seed(SEED0)).to(device)
gan_pils = {}
with torch.no_grad():
    for i, c in enumerate(AUG_CLASSES):
        out = G(z, torch.full((N,), i, device=device, dtype=torch.long)).cpu()
        arr = ((out.permute(0, 2, 3, 1).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        gan_pils[c] = [Image.fromarray(a) for a in arr]   # same 8-bit pathway as headline
results["gan"] = evaluate(as_tensor(gan_pils), real_stats)
save_grid(gan_pils, FIG / "inspect_gan_final.png")
del G
torch.cuda.empty_cache()


# --- arm 2: diffusion --------------------------------------------------------

from diffusers import StableDiffusionPipeline
from peft import PeftModel

from src.diffusion import BASE

samp = cfg["diffusion"]["sampler"]
pipe = StableDiffusionPipeline.from_pretrained(
    BASE, torch_dtype=torch.float16, safety_checker=None).to(device)
pipe.safety_checker = None
pipe.set_progress_bar_config(disable=True)
pipe.enable_attention_slicing()
pipe.vae.enable_slicing()


def diffuse(c, i):
    gen = torch.Generator(device=device).manual_seed(SEED0 + i)
    img = pipe(CLASS_PROMPTS[c], num_inference_steps=samp["num_inference_steps"],
               guidance_scale=samp["guidance_scale"], height=512, width=512,
               generator=gen).images[0]
    return img.resize((SIZE, SIZE))          # same resize as src/diffusion.py


# Adapter-loading check, part 1: base SD BEFORE the adapter modifies the UNet
print("\nBase SD 1.5 (adapter check) ...")
base_pils = {c: [diffuse(c, i) for i in range(GRID_PER_CLASS)] for c in AUG_CLASSES}
save_grid(base_pils, FIG / "inspect_base_sd.png")

peft_unet = PeftModel.from_pretrained(pipe.unet, str(CKPT / "lora_final"))
n_lora = sum(p.numel() for n, p in peft_unet.named_parameters() if "lora_" in n)
b_max = max(p.abs().max().item() for n, p in peft_unet.named_parameters()
            if "lora_B" in n)
print(f"[check] LoRA parameters loaded: {n_lora/1e6:.2f}M (expect ~1.59M)")
print(f"[check] max |lora_B| = {b_max:.5f} (0 would mean an untrained adapter)")
assert n_lora > 1e6 and b_max > 0, "adapter missing or untrained"
pipe.unet = peft_unet.merge_and_unload().to(torch.float16)

print(f"SD 1.5 + LoRA: {N} x {len(AUG_CLASSES)} images ...")
t0 = time.perf_counter()
lora_pils = {}
for c in AUG_CLASSES:
    lora_pils[c] = [diffuse(c, i) for i in range(N)]
    print(f"  {c}: done ({(time.perf_counter() - t0) / 60:.1f} min)")
results["diffusion"] = evaluate(as_tensor(lora_pils), real_stats)
save_grid(lora_pils, FIG / "inspect_lora_final.png")

# Adapter-loading check, part 2: did training change the output?
results["adapter_check"] = {"lora_params_M": round(n_lora / 1e6, 3),
                            "max_lora_B": b_max, "pixel_diff_vs_base": {}}
for c in AUG_CLASSES:
    d = np.mean([np.abs(np.asarray(a, float) - np.asarray(b, float)).mean()
                 for a, b in zip(lora_pils[c][:GRID_PER_CLASS], base_pils[c])])
    results["adapter_check"]["pixel_diff_vs_base"][c] = round(float(d), 1)


# --- report ------------------------------------------------------------------

print(f"\nREAL reference ({N} train images per class)")
print(f"{'class':6} {'spread':>7} {'min_pr':>7} {'mean_rgb':>16}")
for c, s in real_stats.items():
    print(f"{c:6} {s['spread']:7.4f} {s['min_pair']:7.4f} {str(s['mean_rgb']):>16}")
print_table("GAN (gan_final)", results["gan"])
print_table("DIFFUSION (lora_final)", results["diffusion"])
print("\nAdapter check - mean |LoRA - base| pixel difference (0-255):",
      results["adapter_check"]["pixel_diff_vs_base"])

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nSaved {OUT_JSON}")
