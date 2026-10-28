"""
src/fid.py - Frechet Inception Distance per class and pooled (Heusel et al., 2017).

FID = ||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 (S_r S_g)^(1/2)), Inception-v3 pool3
features (2,048-d), via the reference implementation in `pytorch-fid`.

Fixed choices (D25):
  * Real reference = the generator source images of that class (training
    partition, D14); pooled = all 1,526. No test or validation image is used.
  * Every image passes through the same 128 px pathway the classifier receives
    (real images resized, synthetic images already 128), then pytorch-fid
    resizes to 299 internally. Both arms are therefore treated identically.
  * Equal synthetic counts in every comparison: 1,000 per class per arm,
    5,000 pooled per arm.

Limitations stated in advance (METHODS_REFERENCE Part 9): Inception features
come from ImageNet, so FID is indicative for dermatoscopy, not authoritative;
and FID is biased upward when a covariance is estimated from few images. With
71 real DF images the per-class value for DF is unreliable in absolute terms;
the comparison between arms remains valid because both use the same real
reference and the same synthetic count.
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

SIZE = 128
_TF = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor()])   # [0, 1]; pytorch-fid normalises


class InceptionFeatures:
    def __init__(self, device="cuda"):
        from pytorch_fid.inception import InceptionV3
        block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block], resize_input=True,
                                 normalize_input=True).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, paths, batch=100):
        out = []
        for i in range(0, len(paths), batch):
            x = torch.stack([_TF(Image.open(p).convert("RGB"))
                             for p in paths[i:i + batch]]).to(self.device)
            out.append(self.model(x)[0].squeeze(-1).squeeze(-1).cpu().numpy())
        return np.concatenate(out).astype(np.float64)


def stats(feats):
    return feats.mean(0), np.cov(feats, rowvar=False)


def fid(feats_a, feats_b):
    from pytorch_fid.fid_score import calculate_frechet_distance
    mu_a, s_a = stats(feats_a)
    mu_b, s_b = stats(feats_b)
    return float(calculate_frechet_distance(mu_a, s_a, mu_b, s_b))
