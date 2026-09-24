"""
sampling_pilot.py - measure diffusion sampling cost before committing to a schedule.

Why this exists
---------------
Every schedule estimate so far has been a guess, and both guesses were wrong in
the same direction. Diffusion sampling at 512 px is the single largest GPU cost
in Run 1 - larger than every classifier run combined - because the efficiency
curve (D9) needs its own synthetic images at each of four checkpoints, for both
arms, across five classes.

Ten minutes of measurement replaces all of it. The number this prints goes
straight into DECISION_LOG.md (D16) and drives the Kaggle week split.

Run this ON THE PLATFORM YOU WILL SAMPLE ON. A P100 and an RTX 4060 differ
enough that a figure measured on one does not transfer to the other, so run it
once per execution environment.

    python sampling_pilot.py                     # base SD1.5, no LoRA
    python sampling_pilot.py --lora results/checkpoints/lora_unet
    python sampling_pilot.py --n 20 --batch 8 --steps 30

Sampling is EXCLUDED from the declared training budget (D2), but its cost is
reported separately, so this measurement is itself a reportable result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


CLASS_PROMPTS = {
    "akiec": "a dermatoscopic image of an actinic keratosis skin lesion",
    "bcc":   "a dermatoscopic image of a basal cell carcinoma skin lesion",
    "df":    "a dermatoscopic image of a dermatofibroma skin lesion",
    "mel":   "a dermatoscopic image of a melanoma skin lesion",
    "vasc":  "a dermatoscopic image of a vascular skin lesion",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--lora", default=None,
                    help="path to a trained LoRA adapter (optional; timing is "
                         "essentially identical with and without)")
    ap.add_argument("--n", type=int, default=20, help="images to time")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30,
                    help="denoising steps; keep at the declared 30")
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default="logs/sampling_pilot.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. Run this on the sampling platform.")

    from diffusers import StableDiffusionPipeline

    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[pilot] {gpu} ({vram:.1f} GB)")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base, torch_dtype=torch.float16, safety_checker=None,
    ).to(device)
    # Medical images trip the NSFW filter; disabling is necessary, not a shortcut.
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)

    if args.lora:
        pipe.unet.load_attn_procs(args.lora)
        print(f"[pilot] LoRA loaded from {args.lora}")

    if vram < 12:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        print("[pilot] attention and VAE slicing enabled (under 12 GB)")

    prompt = CLASS_PROMPTS["mel"]
    gen = torch.Generator(device=device).manual_seed(42)

    # Warm up: the first call compiles kernels and allocates, and would
    # otherwise inflate the measurement.
    print("[pilot] warming up...")
    pipe([prompt] * min(2, args.batch), num_inference_steps=args.steps,
         guidance_scale=args.guidance, height=args.size, width=args.size,
         generator=gen)
    torch.cuda.synchronize()

    print(f"[pilot] timing {args.n} images at batch {args.batch}, "
          f"{args.steps} steps, {args.size}px...")
    produced, t0 = 0, time.perf_counter()
    while produced < args.n:
        b = min(args.batch, args.n - produced)
        pipe([prompt] * b, num_inference_steps=args.steps,
             guidance_scale=args.guidance, height=args.size, width=args.size,
             generator=gen)
        produced += b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    per_image = elapsed / produced
    peak = torch.cuda.max_memory_allocated() / 1e9

    # Projection against the committed plan.
    headline = 5 * 1000          # 5 classes x headline_per_class (D7)
    curve = 4 * 5 * 500          # 4 checkpoints x 5 classes x curve_per_class (D16)
    curve_at_1000 = 4 * 5 * 1000

    result = {
        "gpu": gpu,
        "vram_gb": round(vram, 2),
        "batch_size": args.batch,
        "num_inference_steps": args.steps,
        "resolution": args.size,
        "lora": args.lora,
        "images_timed": produced,
        "elapsed_s": round(elapsed, 2),
        "seconds_per_image": round(per_image, 3),
        "peak_vram_gb": round(peak, 2),
        "projected_hours": {
            "headline_5000_images": round(headline * per_image / 3600, 2),
            "curve_10000_images_at_500": round(curve * per_image / 3600, 2),
            "curve_20000_images_at_1000": round(curve_at_1000 * per_image / 3600, 2),
        },
    }

    print("\n" + "=" * 62)
    print(f"  {per_image:.2f} s/image   (peak {peak:.2f} GB)")
    print("=" * 62)
    print(f"  Headline sampling, 5,000 images        : "
          f"{result['projected_hours']['headline_5000_images']:6.2f} GPU-h")
    print(f"  Efficiency curve at 500/class          : "
          f"{result['projected_hours']['curve_10000_images_at_500']:6.2f} GPU-h")
    print(f"  Efficiency curve at 1,000/class        : "
          f"{result['projected_hours']['curve_20000_images_at_1000']:6.2f} GPU-h")
    total = (result["projected_hours"]["headline_5000_images"]
             + result["projected_hours"]["curve_10000_images_at_500"])
    print(f"  Diffusion sampling total (D16 plan)    : {total:6.2f} GPU-h")
    print("=" * 62)
    if total > 20:
        print("  Over 20 GPU-h on sampling alone. Consider the declared "
              "fallback: DPMSolverMultistep at 25 steps, applied identically\n"
              "  to every sampling call in both executions.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {args.out}. Record seconds_per_image in DECISION_LOG.md "
          f"under D16 - the schedule derives from it.")


if __name__ == "__main__":
    main()
