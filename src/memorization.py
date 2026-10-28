"""
src/memorization.py - nearest-neighbour memorization audit (proposal Eq. 7.5; D22, D24).

For every synthetic image s, d(s) = min over the generator source images x of
LPIPS(s, x). The near-duplicate threshold is the 1st percentile of the
real-to-real nearest-neighbour distances, where a real image's neighbour is
searched only among real images of OTHER lesions (D22). One pooled threshold,
over the 1,526 generator source images (D24).

Efficiency, and why it is exact (METHODS_REFERENCE 10.2): LPIPS is a weighted
squared Euclidean distance in a fixed, channel-normalised feature space, so

    phi(x) = concat_l  sqrt(w_l) * normalise(f_l(x)) / sqrt(H_l * W_l)

gives ||phi(a) - phi(b)||^2 == LPIPS(a, b). Each image is embedded once and all
distances come from matrix products. `verify_equivalence` checks this against
the lpips package directly and must pass before any audit result is used.

Distances reported are LPIPS values (squared Euclidean in phi-space). Equation
7.5 writes the unsquared norm; the nearest neighbour and the percentile rank
are identical under either, so no result depends on the choice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

SIZE = 128     # the resolution the classifier receives (D24)
_TF = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(),
                 T.Normalize([0.5] * 3, [0.5] * 3)])   # LPIPS expects [-1, 1]


def load_images(paths):
    """Same resize pathway for real and synthetic images (synthetic are already 128)."""
    return torch.stack([_TF(Image.open(p).convert("RGB")) for p in paths])


class LPIPSEmbedder:
    """Embeds images so that squared Euclidean distance equals LPIPS (AlexNet, v0.1)."""

    def __init__(self, device="cuda"):
        import lpips
        self.lpips = lpips
        self.net = lpips.LPIPS(net="alex", version="0.1", verbose=False).to(device).eval()
        self.device = device
        self.sqrt_w = []
        for lin in self.net.lins:
            conv = [m for m in lin.model.modules() if isinstance(m, nn.Conv2d)][0]
            w = conv.weight.detach().view(-1)
            assert (w >= 0).all(), "LPIPS linear weights must be non-negative"
            self.sqrt_w.append(w.sqrt().view(1, -1, 1, 1))

    @torch.no_grad()
    def embed(self, x):
        """x in [-1, 1], shape [n, 3, H, W] -> [n, D] float32."""
        x = self.net.scaling_layer(x.to(self.device))
        feats = self.net.net.forward(x)
        parts = []
        for k, f in enumerate(feats):
            f = self.lpips.normalize_tensor(f)
            h, w = f.shape[2], f.shape[3]
            parts.append((f * self.sqrt_w[k] / (h * w) ** 0.5).flatten(1))
        return torch.cat(parts, dim=1).float()

    @torch.no_grad()
    def verify_equivalence(self, a, b, tol=1e-3):
        """Mandatory check: embedding distance must reproduce LPIPS directly."""
        direct = self.net(a.to(self.device), b.to(self.device)).view(-1)
        ea, eb = self.embed(a), self.embed(b)
        # Check the exact code path the audit uses (matrix-product distances).
        via = _sq_dists(ea, eb, (eb ** 2).sum(1)).diagonal()
        err = (direct - via).abs().max().item()
        print(f"[audit] LPIPS equivalence: max |direct - embedded| = {err:.2e} "
              f"over {len(a)} pairs (tolerance {tol})")
        assert err < tol, "embedding does not reproduce LPIPS - audit invalid"
        return err


def embed_paths(emb, paths, batch=64):
    out = []
    for i in range(0, len(paths), batch):
        out.append(emb.embed(load_images(paths[i:i + batch])))
    return torch.cat(out)


def _sq_dists(a, b, b_sq):
    """Squared Euclidean distances, [len(a), len(b)], clamped at zero."""
    a_sq = (a ** 2).sum(1, keepdim=True)
    return (a_sq + b_sq[None, :] - 2.0 * a @ b.T).clamp_min_(0)


def real_to_real(ref_emb, lesion_ids, chunk=128):
    """Nearest neighbour of each real image among real images of OTHER lesions (D22)."""
    les = np.asarray(lesion_ids)
    codes = torch.as_tensor(pd.factorize(les)[0], device=ref_emb.device)
    b_sq = (ref_emb ** 2).sum(1)
    dmin, amin = [], []
    for i in range(0, len(ref_emb), chunk):
        d = _sq_dists(ref_emb[i:i + chunk], ref_emb, b_sq)
        same = codes[i:i + chunk, None] == codes[None, :]   # includes self
        d[same] = float("inf")
        v, j = d.min(1)
        dmin.append(v.cpu()); amin.append(j.cpu())
    return torch.cat(dmin).numpy(), torch.cat(amin).numpy()


def synthetic_to_real(emb, syn_paths, ref_emb, batch=64):
    """d(s) for each synthetic image, Eq. 7.5, and the index of its nearest real image."""
    b_sq = (ref_emb ** 2).sum(1)
    dmin, amin = [], []
    for i in range(0, len(syn_paths), batch):
        e = emb.embed(load_images(syn_paths[i:i + batch]))
        v, j = _sq_dists(e, ref_emb, b_sq).min(1)
        dmin.append(v.cpu()); amin.append(j.cpu())
    return torch.cat(dmin).numpy(), torch.cat(amin).numpy()


def calibrate_threshold(r2r, percentile=1.0):
    """tau = 1st percentile of the real-to-real (other-lesion) distribution (D22, D24)."""
    tau = float(np.percentile(r2r, percentile))
    print(f"[audit] threshold tau = {tau:.4f} "
          f"({percentile}th percentile of {len(r2r)} real-to-real distances)")
    return tau


def summarise(df, tau):
    """Near-duplicate rate per arm and class, plus pooled, and distance quantiles."""
    rows = []
    for (arm, cls), g in list(df.groupby(["arm", "dx"])) + \
            [((a, "ALL"), g) for a, g in df.groupby("arm")]:
        d = g["nn_dist"].to_numpy()
        rows.append({"arm": arm, "class": cls, "n": len(d),
                     "near_duplicates": int((d < tau).sum()),
                     "rate": float((d < tau).mean()),
                     "min": d.min(), "p01": np.percentile(d, 1),
                     "p05": np.percentile(d, 5), "median": np.median(d)})
    return pd.DataFrame(rows)


def save_worst_pairs(df, ref_paths, out_png, n=10):
    """The n closest synthetic-real pairs per arm, side by side (proposal 7.5).

    Contains real patient images: saved outside version control and never
    published (proposal 9.1).
    """
    tiles = []
    for arm, g in df.groupby("arm"):
        for _, r in g.nsmallest(n, "nn_dist").iterrows():
            s = Image.open(r["path"]).convert("RGB").resize((SIZE, SIZE))
            x = Image.open(ref_paths[int(r["nn_index"])]).convert("RGB").resize((SIZE, SIZE))
            tiles.append((s, x))
    grid = Image.new("RGB", (SIZE * 2 * n, SIZE * (len(tiles) // n)), "white")
    for k, (s, x) in enumerate(tiles):
        row, col = divmod(k, n)
        grid.paste(s, (col * 2 * SIZE, row * SIZE))
        grid.paste(x, (col * 2 * SIZE + SIZE, row * SIZE))
    grid.save(out_png)
    print(f"[audit] saved {out_png} (rows: arms in alphabetical order; "
          f"each pair = synthetic | nearest real)")
