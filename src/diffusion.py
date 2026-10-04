"""
src/diffusion.py - LoRA fine-tuning of Stable Diffusion 1.5 for minority classes.

Trained at 512 px (D6): the SD VAE downsamples by 8, so a 128 px input would
give 16x16 latents against the 64x64 the UNet expects. Generated images are
downsampled to 128 px before the classifier sees them, so every condition is
evaluated at the same resolution.

The budget covers adaptation to HAM10000 only. Pretraining is excluded and
treated as a freely available resource of the same status as the ImageNet
weights used by the classifier (PRE_REGISTRATION 3.1).
"""

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer

BASE = "runwayml/stable-diffusion-v1-5"


def build_lora_unet(rank=8, alpha=16, base=BASE):
    """Freeze SD 1.5's UNet and attach low-rank adapters to its attention
    projections (Hu et al., 2021 - proposal Equation 7.4).

    W = W_0 + (alpha / r) * B @ A, with W_0 frozen and only A, B trained.
    Full fine-tuning would exhaust the declared budget in minutes and overfit
    1,526 images, making the equal-budget comparison impossible.
    """
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet")
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, cfg)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"[lora] trainable {trainable/1e6:.2f}M of {total/1e6:.1f}M "
          f"({100*trainable/total:.3f}%)")
    if trainable / total > 0.05:
        raise RuntimeError(
            "LoRA did not attach: more than 5% of parameters are trainable. "
            "This would be a full fine-tune, which invalidates the equal-budget "
            "comparison. Check that target_modules match the UNet's attention "
            "layer names."
        )
    return unet


def load_frozen_components(base=BASE, device="cuda"):
    """VAE, text encoder and tokenizer - all frozen, none trained."""
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    vae.requires_grad_(False).to(device).eval()
    text_encoder.requires_grad_(False).to(device).eval()
    return vae, text_encoder, tokenizer, noise_scheduler
import torch.nn.functional as F
from pathlib import Path

from src.data import AUG_CLASSES, CLASS_PROMPTS


def train_lora(unet, vae, text_encoder, tokenizer, noise_scheduler, loader, cfg,
               tracker, device="cuda", save_every_minutes=10.0,
               out_dir="results", grad_accum=None):
    """Minimise noise-prediction error (proposal Equation 7.2).

    The model is shown a noised latent and asked which noise was added. The
    loss is a plain MSE against freshly sampled noise at a random timestep,
    so it is inherently noisy - judge progress over hundreds of steps, never
    step to step.
    """
    from src.checkpointing import save_state

    grad_accum = grad_accum or cfg.get("grad_accum", 8)
    unet.to(device).train()
    if cfg.get("gradient_checkpointing", True):
        unet.enable_gradient_checkpointing()

    params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"])
    scaler = torch.amp.GradScaler("cuda")

    out = Path(out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    micro, stop = 0, False
    while not stop:
        for imgs, prompts in loader:
            imgs = imgs.to(device)

            with torch.no_grad():
                # Encode to latent space (Rombach et al., 2022)
                latents = vae.encode(imgs).latent_dist.sample() * vae.config.scaling_factor
                tok = tokenizer(list(prompts), padding="max_length",
                                max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors="pt").input_ids.to(device)
                cond = text_encoder(tok)[0]

            noise = torch.randn_like(latents)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                              (latents.size(0),), device=device).long()
            noisy = noise_scheduler.add_noise(latents, noise, t)    # Equation 7.3

            with torch.amp.autocast("cuda"):
                pred = unet(noisy, t, encoder_hidden_states=cond).sample
                loss = F.mse_loss(pred.float(), noise.float())      # Equation 7.2

            scaler.scale(loss / grad_accum).backward()
            micro += 1

            # One optimiser step per grad_accum micro-batches; the tracker counts
            # optimiser steps, not micro-batches, so the figure means what the
            # dissertation says it means.
            if micro % grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                tracker.record_step(loss=round(loss.item(), 4))

                label = tracker.checkpoint_due()
                if label:
                    with tracker.timed_save():
                        unet.save_pretrained(out / "checkpoints" / f"lora_{label}")
                        save_state(out / "checkpoints" / f"lora_{label}_meta.pt",
                                   tracker.steps, tracker, {}, {"opt": opt})

                if tracker.save_due(every_minutes=save_every_minutes):
                    with tracker.timed_save():
                        unet.save_pretrained(out / "checkpoints" / "lora_resume")
                        save_state(out / "checkpoints" / "lora_resume_meta.pt",
                                   tracker.steps, tracker, {}, {"opt": opt})

                if tracker.exhausted() or tracker.session_expiring():
                    stop = True
                    break

    with tracker.timed_save():
        unet.save_pretrained(out / "checkpoints" / "lora_final")
        save_state(out / "checkpoints" / "lora_final_meta.pt",
                   tracker.steps, tracker, {}, {"opt": opt})
    return unet, tracker.history
from diffusers import StableDiffusionPipeline


def build_pipeline(unet, base=BASE, device="cuda"):
    """Assemble a sampling pipeline around the fine-tuned UNet.

    The safety checker is disabled: dermatoscopic images trigger the NSFW
    filter, which would silently discard valid samples. No generated image is
    exposed publicly. To be stated in the methodology chapter.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        base, unet=unet, torch_dtype=torch.float16, safety_checker=None,
    ).to(device)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)

    if torch.cuda.get_device_properties(0).total_memory / 1e9 < 12:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    return pipe


def generate_diffusion(pipe, classes, n_per_class, out_dir, base_seed=42,
                       steps=30, guidance=7.5, size=512, target_size=128,
                       manifest_path=None, session_limit_hours=None):
    """Synthesise images, resumable and deterministic per image.

    Each image's seed is derived from (base_seed, class, index), so an
    interrupted run resumes to byte-identical output rather than producing a
    different set depending on how often the session died.

    Generated at 512 px (the UNet's native resolution) and downsampled to
    128 px, so the classifier sees the same resolution in every condition.
    """
    from src.checkpointing import resumable_generate

    def one(cls, gen):
        img = pipe(CLASS_PROMPTS[cls], num_inference_steps=steps,
                   guidance_scale=guidance, height=size, width=size,
                   generator=gen).images[0]
        return img.resize((target_size, target_size))

    return resumable_generate(
        one, classes, n_per_class, out_dir, base_seed=base_seed,
        manifest_path=manifest_path, session_limit_hours=session_limit_hours,
    )


@torch.no_grad()
def generate_gan_images(G, classes, n_per_class, out_dir, latent_dim=128,
                        base_seed=42, device="cuda", manifest_path=None):
    """GAN counterpart, using the same deterministic per-image seeding."""
    from PIL import Image
    import numpy as np
    from src.checkpointing import resumable_generate
    from src.data import AUG_C2I

    G.to(device).eval()

    def one(cls, gen):
        z = torch.randn(1, latent_dim, device=device, generator=gen)
        y = torch.tensor([AUG_C2I[cls]], device=device)
        img = G(z, y)[0].cpu()                      # [-1, 1]
        arr = ((img.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    return resumable_generate(
        one, classes, n_per_class, out_dir, base_seed=base_seed,
        manifest_path=manifest_path,
    )