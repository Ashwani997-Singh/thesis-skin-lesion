"""
gpu_probe.py - hardware feasibility probe for the equal-budget thesis pipeline.

Measures, on YOUR machine rather than by estimate:
  1. Device, VRAM and enforced power limit
  2. Whether the GPU throttles under sustained load (critical: your budget runs
     are 4 hours long, and both arms must throttle identically)
  3. Peak VRAM and seconds/step for a ResNet-50 training step
  4. Peak VRAM and seconds/step for a WGAN-GP step at batch 32 and 64,
     including the gradient-penalty double-backward that dominates memory
  5. A projection of how many steps a 4.0 GPU-hour budget actually buys

Run with the laptop on MAINS POWER and the vendor performance profile you
intend to use for the real runs. Close other GPU applications first.

    python gpu_probe.py                # default 60s soak test
    python gpu_probe.py --soak 300     # 5 min soak, better throttle signal
    python gpu_probe.py --soak 0       # skip soak

Requires only torch and torchvision. Not yet run in your environment, so
expect to fix version differences.
"""

import argparse
import json
import subprocess
import time

import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Device information
# --------------------------------------------------------------------------


def nvidia_smi_query(fields):
    """Query nvidia-smi. Returns a dict, or None if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        values = [v.strip() for v in out.stdout.strip().split("\n")[0].split(",")]
        return dict(zip(fields, values))
    except Exception:
        return None


def device_report():
    print("=" * 68)
    print("DEVICE")
    print("=" * 68)
    if not torch.cuda.is_available():
        print("CUDA not available. Check your PyTorch install (need a cu12x build).")
        raise SystemExit(1)

    props = torch.cuda.get_device_properties(0)
    print(f"  GPU              : {props.name}")
    print(f"  Total VRAM       : {props.total_memory / 1e9:.2f} GB")
    print(f"  Compute capability: {props.major}.{props.minor}")
    print(f"  PyTorch          : {torch.__version__}")
    print(f"  CUDA (torch)     : {torch.version.cuda}")

    smi = nvidia_smi_query(
        ["power.limit", "power.max_limit", "clocks.max.sm", "temperature.gpu"]
    )
    if smi:
        print(f"  Power limit      : {smi['power.limit']} W "
              f"(hardware max {smi['power.max_limit']} W)")
        print(f"  Max SM clock     : {smi['clocks.max.sm']} MHz")
        print(f"  Idle temperature : {smi['temperature.gpu']} C")
        try:
            if float(smi["power.max_limit"]) - float(smi["power.limit"]) > 5:
                print("  NOTE: enforced limit is below hardware max. Dynamic Boost "
                      "may raise it under load - the soak test will show whether "
                      "it does.")
        except ValueError:
            pass
    print()


# --------------------------------------------------------------------------
# Models mirroring the thesis pipeline
# --------------------------------------------------------------------------


class Generator(nn.Module):
    """Class-conditional DCGAN-style generator, 128x128 output."""

    def __init__(self, latent_dim=128, n_classes=7, ngf=64):
        super().__init__()
        self.embed = nn.Embedding(n_classes, latent_dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False), nn.Tanh(),
        )

    def forward(self, z, labels):
        z = torch.cat([z, self.embed(labels)], dim=1)
        return self.net(z.unsqueeze(-1).unsqueeze(-1))


class Critic(nn.Module):
    """No BatchNorm - it breaks the per-sample gradient penalty."""

    def __init__(self, n_classes=7, ndf=64, img_size=128):
        super().__init__()
        self.img_size = img_size
        self.embed = nn.Embedding(n_classes, img_size * img_size)
        self.net = nn.Sequential(
            nn.Conv2d(4, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 2, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 4, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 8, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 16, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 16, 1, 4, 1, 0),
        )

    def forward(self, x, labels):
        emb = self.embed(labels).view(-1, 1, self.img_size, self.img_size)
        return self.net(torch.cat([x, emb], dim=1)).view(-1)


def gradient_penalty(critic, real, fake, labels, device):
    eps = torch.rand(real.size(0), 1, 1, 1, device=device)
    x_int = (eps * real + (1 - eps) * fake).requires_grad_(True)
    scores = critic(x_int, labels)
    grads = torch.autograd.grad(
        outputs=scores, inputs=x_int,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True,
    )[0].view(real.size(0), -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------


def bench_classifier(device, batch_size=32, size=128, iters=20):
    """ResNet-50 training step with AMP, as in Conditions A-D."""
    from torchvision import models

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    x = torch.randn(batch_size, 3, size, size, device=device)
    y = torch.randint(0, 7, (batch_size,), device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = crit(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters

    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, opt
    torch.cuda.empty_cache()
    return elapsed, peak


def bench_wgan_gp(device, batch_size, size=128, n_critic=5, iters=5):
    """One full WGAN-GP generator step = n_critic critic updates + 1 G update.

    Run in fp32: the gradient penalty is unreliable under autocast, which is
    why the guide's GAN loop does not use AMP.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    G = Generator().to(device)
    C = Critic(img_size=size).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9))
    optC = torch.optim.Adam(C.parameters(), lr=2e-4, betas=(0.5, 0.9))

    real = torch.randn(batch_size, 3, size, size, device=device)
    labels = torch.randint(0, 7, (batch_size,), device=device)

    def step():
        for _ in range(n_critic):
            z = torch.randn(batch_size, 128, device=device)
            fake = G(z, labels).detach()
            gp = gradient_penalty(C, real, fake, labels, device)
            lossC = C(fake, labels).mean() - C(real, labels).mean() + 10.0 * gp
            optC.zero_grad(set_to_none=True)
            lossC.backward()
            optC.step()
        z = torch.randn(batch_size, 128, device=device)
        lossG = -C(G(z, labels), labels).mean()
        optG.zero_grad(set_to_none=True)
        lossG.backward()
        optG.step()

    for _ in range(2):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters

    peak = torch.cuda.max_memory_allocated() / 1e9
    del G, C, optG, optC
    torch.cuda.empty_cache()
    return elapsed, peak


def soak_test(device, seconds, size=128, batch_size=32):
    """Hold the GPU under load and sample power, clocks and temperature.

    A falling clock or power figure over time means throttling. That is not
    fatal - but it must be the SAME for both generative arms, so record it.
    """
    print("=" * 68)
    print(f"SOAK TEST ({seconds}s under sustained WGAN-GP load)")
    print("=" * 68)

    G = Generator().to(device)
    C = Critic(img_size=size).to(device)
    optC = torch.optim.Adam(C.parameters(), lr=2e-4, betas=(0.5, 0.9))
    labels = torch.randint(0, 7, (batch_size,), device=device)
    real = torch.randn(batch_size, 3, size, size, device=device)

    fields = ["power.draw", "clocks.sm", "temperature.gpu", "utilization.gpu"]
    samples = []
    t_start = time.perf_counter()
    next_sample = 0.0

    print(f"  {'t(s)':>6} {'power(W)':>9} {'SM(MHz)':>8} {'temp(C)':>8} {'util(%)':>8}")
    while (t := time.perf_counter() - t_start) < seconds:
        z = torch.randn(batch_size, 128, device=device)
        fake = G(z, labels).detach()
        gp = gradient_penalty(C, real, fake, labels, device)
        lossC = C(fake, labels).mean() - C(real, labels).mean() + 10.0 * gp
        optC.zero_grad(set_to_none=True)
        lossC.backward()
        optC.step()

        if t >= next_sample:
            smi = nvidia_smi_query(fields)
            if smi:
                samples.append({"t": round(t, 1), **smi})
                print(f"  {t:6.0f} {smi['power.draw']:>9} {smi['clocks.sm']:>8} "
                      f"{smi['temperature.gpu']:>8} {smi['utilization.gpu']:>8}")
            next_sample = t + 15.0

    del G, C, optC
    torch.cuda.empty_cache()

    if len(samples) >= 2:
        try:
            first = float(samples[0]["clocks.sm"])
            last = float(samples[-1]["clocks.sm"])
            drop = (first - last) / first * 100 if first else 0.0
            print(f"\n  SM clock drift over soak: {drop:+.1f}%")
            if drop > 10:
                print("  -> Throttling under sustained load. Acceptable, but keep "
                      "conditions identical across both arms and state it.")
            else:
                print("  -> Clocks stable. Good news for a 4-hour budgeted run.")
            peak_power = max(float(s["power.draw"]) for s in samples)
            print(f"  Peak power draw observed: {peak_power:.1f} W")
        except ValueError:
            pass
    print()
    return samples


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soak", type=int, default=60,
                    help="seconds of sustained load for the throttle test (0 to skip)")
    ap.add_argument("--budget-hours", type=float, default=4.0,
                    help="declared per-arm budget, for the step projection")
    ap.add_argument("--out", default="gpu_probe_results.json")
    args = ap.parse_args()

    device_report()
    device = torch.device("cuda")
    results = {}

    print("=" * 68)
    print("THROUGHPUT AND PEAK VRAM")
    print("=" * 68)

    t, mem = bench_classifier(device)
    results["resnet50_bs32"] = {"sec_per_step": t, "peak_vram_gb": mem}
    print(f"  ResNet-50 @128 bs32 (AMP) : {t*1000:7.1f} ms/step   peak {mem:.2f} GB")

    for bs in (32, 64):
        try:
            t, mem = bench_wgan_gp(device, batch_size=bs)
            results[f"wgan_gp_bs{bs}"] = {"sec_per_step": t, "peak_vram_gb": mem}
            steps = int(args.budget_hours * 3600 / t)
            print(f"  WGAN-GP  @128 bs{bs:<3}(fp32): {t*1000:7.1f} ms/step   "
                  f"peak {mem:.2f} GB   -> ~{steps:,} G-steps in "
                  f"{args.budget_hours}h")
        except torch.cuda.OutOfMemoryError:
            results[f"wgan_gp_bs{bs}"] = {"oom": True}
            print(f"  WGAN-GP  @128 bs{bs:<3}(fp32): OUT OF MEMORY - drop to a "
                  f"smaller batch")
            torch.cuda.empty_cache()

    print()
    if args.soak > 0:
        results["soak"] = soak_test(device, args.soak)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.out} - keep this; it is evidence for your "
          f"budget-control section.")


if __name__ == "__main__":
    main()
