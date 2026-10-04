"""
src/gan.py - conditional WGAN-GP for minority-class synthesis.

Label space is AUG_CLASSES (5 classes), NOT the classifier's CLASSES (7).
See D14: sizing the embedding at 7 while feeding only 5 labels would leave two
rows untrained and cause labels passed at generation time to index the wrong
row - a failure that produces images without raising an error.
"""

import torch
import torch.nn as nn

from src.data import AUG_CLASSES

N_CLASSES = len(AUG_CLASSES)      # 5, derived - never hard-coded


class Generator(nn.Module):
    """Class-conditional DCGAN-style generator, 128x128 output."""

    def __init__(self, latent_dim=128, n_classes=N_CLASSES, ngf=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed = nn.Embedding(n_classes, latent_dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16), nn.ReLU(True),          # 4x4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),           # 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),           # 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),           # 32x32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),               # 64x64
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),                                        # 128x128, [-1,1]
        )

    def forward(self, z, labels):
        z = torch.cat([z, self.embed(labels)], dim=1)
        return self.net(z.unsqueeze(-1).unsqueeze(-1))


class Critic(nn.Module):
    """Class-conditional critic. Outputs a score, not a probability.

    InstanceNorm, never BatchNorm: BatchNorm makes each output depend on the
    whole batch, so the gradient of the critic with respect to one input is no
    longer a per-sample gradient, and the gradient penalty stops enforcing a
    per-sample Lipschitz constraint.
    """

    def __init__(self, n_classes=N_CLASSES, ndf=64, img_size=128):
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
    """Proposal Equation 7.1, penalty term.

    A differentiable function is 1-Lipschitz exactly when its gradient has norm
    at most 1 everywhere. Rather than clipping weights, which costs capacity,
    the deviation from unit gradient norm is penalised directly at points
    interpolated between real and generated samples (Gulrajani et al., 2017).
    """
    eps = torch.rand(real.size(0), 1, 1, 1, device=device)
    x_int = (eps * real + (1 - eps) * fake).requires_grad_(True)
    scores = critic(x_int, labels)
    grads = torch.autograd.grad(
        outputs=scores, inputs=x_int,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True,
    )[0]
    grads = grads.view(grads.size(0), -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def critic_loss(critic, real, fake, labels, device, lambda_gp=10.0):
    """L_D = E[D(fake)] - E[D(real)] + lambda * GP   (Equation 7.1)

    Note the sign: the critic maximises the gap between real and fake scores,
    so minimising this loss widens it. -critic_loss (excluding the penalty) is
    an estimate of the Wasserstein distance, which is why the WGAN loss curve
    is readable in a way the original GAN's is not.
    """
    gp = gradient_penalty(critic, real, fake, labels, device)
    return critic(fake, labels).mean() - critic(real, labels).mean() + lambda_gp * gp, gp


def generator_loss(critic, fake, labels):
    """L_G = -E[D(fake)] - the generator tries to raise the critic's score."""
    return -critic(fake, labels).mean()
import torchvision.utils as vutils
from pathlib import Path


def train_gan(G, C, loader, cfg, tracker, device="cuda",
              sample_every_minutes=10.0, save_every_minutes=10.0,
              out_dir="results", fixed_noise=None):
    """Train the conditional WGAN-GP until the declared budget is exhausted.

    The budget, not an epoch count, decides when training stops. Non-convergence
    within budget is a finding, not a fault to be corrected (proposal 7.3).
    """
    from src.checkpointing import save_state

    G.to(device).train()
    C.to(device).train()
    optG = torch.optim.Adam(G.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]))
    optC = torch.optim.Adam(C.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]))

    out = Path(out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # Fixed noise so sample grids across checkpoints are directly comparable
    if fixed_noise is None:
        g = torch.Generator(device=device).manual_seed(123)
        fixed_noise = (
            torch.randn(N_CLASSES * 4, cfg["latent_dim"], device=device, generator=g),
            torch.arange(N_CLASSES, device=device).repeat_interleave(4),
        )

    history, stop = [], False
    while not stop:
        for real, labels in loader:
            real, labels = real.to(device), labels.to(device)
            bs = real.size(0)

            # --- critic: n_critic updates per generator update ---
            for _ in range(cfg["n_critic"]):
                z = torch.randn(bs, cfg["latent_dim"], device=device)
                fake = G(z, labels).detach()
                lossC, gp = critic_loss(C, real, fake, labels, device,
                                        lambda_gp=cfg["lambda_gp"])
                optC.zero_grad(set_to_none=True)
                lossC.backward()
                optC.step()

            # --- generator ---
            z = torch.randn(bs, cfg["latent_dim"], device=device)
            lossG = generator_loss(C, G(z, labels), labels)
            optG.zero_grad(set_to_none=True)
            lossG.backward()
            optG.step()

            tracker.record_step(
                lossC=round(lossC.item(), 4),
                lossG=round(lossG.item(), 4),
                gp=round(gp.item(), 4),
                w_est=round(-(lossC.item() - cfg["lambda_gp"] * gp.item()), 4),
            )

            # D9 efficiency-curve checkpoint
            label = tracker.checkpoint_due()
            if label:
                with tracker.timed_save():
                    save_state(out / "checkpoints" / f"gan_{label}.pt",
                               tracker.steps, tracker, {"G": G, "C": C},
                               {"optG": optG, "optC": optC})
                _save_grid(G, fixed_noise, out / "figures" / f"gan_samples_{label}.png")

            # D17 resume checkpoint, scheduled by wall clock
            if tracker.save_due(every_minutes=save_every_minutes):
                with tracker.timed_save():
                    save_state(out / "checkpoints" / "gan_resume.pt",
                               tracker.steps, tracker, {"G": G, "C": C},
                               {"optG": optG, "optC": optC})

            if tracker.exhausted() or tracker.session_expiring():
                stop = True
                break

    with tracker.timed_save():
        save_state(out / "checkpoints" / "gan_final.pt", tracker.steps, tracker,
                   {"G": G, "C": C}, {"optG": optG, "optC": optC})
    _save_grid(G, fixed_noise, out / "figures" / "gan_samples_final.png")
    return G, tracker.history


@torch.no_grad()
def _save_grid(G, fixed, path):
    """Sample grid from fixed noise, so progress across checkpoints is visible."""
    G.eval()
    z, y = fixed
    vutils.save_image(G(z, y).cpu(), str(path), nrow=4,
                      normalize=True, value_range=(-1, 1))
    G.train()