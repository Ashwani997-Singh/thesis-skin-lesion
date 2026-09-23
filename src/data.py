"""
src/data.py - datasets, path indexing and loaders.

Fixes two defects:

1. Image paths were built by string-concatenating a directory, and the Kaggle
   notebook hardcoded ".../HAM10000_images_part_1". The mirror stores images in
   two sibling folders, so roughly 5,000 images were unreachable. Paths are now
   resolved through an index built by recursive glob, which behaves identically
   for the flat local folder and the two-folder Kaggle mirror.

2. The GAN arm trained on all seven classes while the diffusion arm trained on
   the five prompt-covered classes - roughly 7,010 images against 1,550. Under
   an equal-budget design that is an uncontrolled asymmetry in the one factor
   the study exists to hold constant. Both arms now draw on one shared source
   frame (D14), and assert_matched_sources() proves it.

LABEL SPACES - do not mix these up:
    CLASSES     7 classes, used by the CLASSIFIER (all conditions).
    AUG_CLASSES 5 classes, used by BOTH GENERATIVE ARMS (D5, D14).
The GAN's embedding must be sized from AUG_CLASSES, not CLASSES. Sizing it at
7 while only ever feeding 5 labels leaves two rows untrained and makes any
label passed at generation time index the wrong embedding row.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# --- label spaces ----------------------------------------------------------

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
C2I = {c: i for i, c in enumerate(CLASSES)}

AUG_CLASSES = ["akiec", "bcc", "df", "mel", "vasc"]
AUG_C2I = {c: i for i, c in enumerate(AUG_CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# One caption per generated class. Keys must equal AUG_CLASSES exactly.
CLASS_PROMPTS = {
    "akiec": "a dermatoscopic image of an actinic keratosis skin lesion",
    "bcc":   "a dermatoscopic image of a basal cell carcinoma skin lesion",
    "df":    "a dermatoscopic image of a dermatofibroma skin lesion",
    "mel":   "a dermatoscopic image of a melanoma skin lesion",
    "vasc":  "a dermatoscopic image of a vascular skin lesion",
}
assert set(CLASS_PROMPTS) == set(AUG_CLASSES), "prompt keys must match AUG_CLASSES"


# --- path indexing ---------------------------------------------------------

def build_image_index(root, expected=10015, strict=True):
    """Map image_id -> absolute path by recursive glob.

    Handles the flat local folder and the Kaggle mirror's part_1/part_2 layout
    identically, so config.yaml can point at the parent on both platforms.

    The expected-count assertion catches three distinct failure modes in one
    line: an archive that did not fully extract, a wrong path, and a Kaggle
    mirror attached at the wrong level.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image root does not exist: {root}")

    index = {}
    for pattern in ("*.jpg", "*.JPG", "*.jpeg", "*.png"):
        for p in root.rglob(pattern):
            index.setdefault(p.stem, str(p))

    if not index:
        raise FileNotFoundError(f"No images found under {root}")

    if strict and expected is not None and len(index) != expected:
        raise AssertionError(
            f"Found {len(index)} images under {root}, expected {expected}. "
            f"Check that BOTH archives extracted, or that the Kaggle dataset "
            f"is attached at the parent directory rather than one subfolder."
        )

    print(f"[data] indexed {len(index)} images under {root}")
    return index


# --- classifier dataset ----------------------------------------------------

class HAM10000(Dataset):
    """Dermatoscopic images with diagnostic labels.

    Parameters
    ----------
    csv_path : str
        A split CSV, or an augmented CSV containing synthetic rows.
    image_root : str
        Parent directory of the real images. Indexed recursively.
    size : int
        Output resolution. 128 for the classifier and GAN, 512 for diffusion.
    augment : bool
        Classical geometric/photometric augmentation. Condition B only.
        NEVER enable for validation or test (D4).
    class_space : {"full", "augmented"}
        "full" returns indices over the 7 classifier classes.
        "augmented" filters to AUG_CLASSES and returns indices over those 5.
    index : dict, optional
        A prebuilt image index, to avoid re-globbing for every dataset.
    """

    def __init__(self, csv_path=None, image_root=None, size=128, augment=False,
                 class_space="full", index=None, extra_rows=None,
                 expected_index_size=10015, frame=None,
                 normalise="imagenet"):
        if (csv_path is None) == (frame is None):
            raise ValueError("provide exactly one of csv_path or frame")
        self.df = pd.read_csv(csv_path) if csv_path is not None else frame.copy()
        if extra_rows is not None:
            self.df = pd.concat([self.df, extra_rows], ignore_index=True)

        if class_space not in ("full", "augmented"):
            raise ValueError("class_space must be 'full' or 'augmented'")
        self.class_space = class_space
        self.c2i = C2I if class_space == "full" else AUG_C2I

        if class_space == "augmented":
            self.df = self.df[self.df.dx.isin(AUG_CLASSES)].reset_index(drop=True)

        self.index = index if index is not None else build_image_index(
            image_root, expected=expected_index_size
        )

        base = [T.Resize((size, size))]
        aug = [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.05),
            T.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ]
        # "imagenet" for the classifier (matches the pretrained weights).
        # "tanh" maps to [-1, 1] for the GAN, matching the generator's Tanh
        # output; mismatching these is a silent failure, not an error.
        if normalise == "imagenet":
            norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        elif normalise == "tanh":
            norm = T.Normalize([0.5] * 3, [0.5] * 3)
        else:
            raise ValueError("normalise must be 'imagenet' or 'tanh'")
        self.normalise = normalise

        tail = [T.ToTensor(), norm]
        self.tf = T.Compose(base + (aug if augment else []) + tail)

    def __len__(self):
        return len(self.df)

    def _resolve(self, row):
        # Synthetic rows carry an explicit path; real rows resolve by image_id.
        path = row.get("path")
        if isinstance(path, str) and path:
            return path
        image_id = row["image_id"]
        try:
            return self.index[image_id]
        except KeyError:
            raise KeyError(
                f"image_id '{image_id}' not present in the image index. "
                f"The index holds {len(self.index)} entries - if that is not "
                f"10015, the dataset is incompletely extracted or attached."
            )

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self._resolve(row)).convert("RGB")
        return self.tf(img), self.c2i[row["dx"]]


# --- diffusion dataset -----------------------------------------------------

class PromptedLesions(Dataset):
    """Images with class captions, for LoRA fine-tuning.

    Normalised to [-1, 1] as the Stable Diffusion VAE expects, NOT with
    ImageNet statistics. Feeding ImageNet-normalised tensors to the VAE is a
    silent failure: it encodes without error and trains toward the wrong
    distribution.
    """

    def __init__(self, frame, image_root, size=512, index=None):
        self.df = frame.reset_index(drop=True)
        self.index = index if index is not None else build_image_index(image_root)
        self.tf = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = row.get("path")
        if not (isinstance(path, str) and path):
            path = self.index[row["image_id"]]
        img = Image.open(path).convert("RGB")
        return self.tf(img), CLASS_PROMPTS[row["dx"]]


# --- D14: one shared source frame for both arms ----------------------------

def generator_source_frame(train_csv, classes=None):
    """The training rows BOTH generative arms see. Single point of truth.

    Restricted to AUG_CLASSES so the budget is concentrated on the classes
    actually synthesised. Training on all seven would spend roughly two thirds
    of a fixed four-hour budget on melanocytic nevi, which neither arm
    generates.
    """
    classes = list(classes or AUG_CLASSES)
    df = pd.read_csv(train_csv)
    if "synthetic" in df.columns:
        df = df[~df["synthetic"].fillna(False).astype(bool)]
    frame = df[df.dx.isin(classes)].reset_index(drop=True)
    print(f"[data] generator source frame: {len(frame)} images across "
          f"{sorted(frame.dx.unique())}")
    print(frame.dx.value_counts().to_string())
    return frame


def make_generator_loaders(train_csv, image_root, gan_size=128, diff_size=512,
                           gan_batch=64, diff_batch=1, index=None,
                           num_workers=0):
    """Build both generative loaders from ONE frame, and prove they match.

    Returns (gan_loader, diffusion_loader, frame).
    """
    frame = generator_source_frame(train_csv)
    index = index or build_image_index(image_root)

    gan_ds = HAM10000(
        frame=frame,
        image_root=image_root,
        size=gan_size,
        augment=False,              # the GAN learns the real distribution
        class_space="augmented",    # 5-class label space (D14)
        normalise="tanh",           # [-1, 1], matching the generator's Tanh
        index=index,
    )

    diff_ds = PromptedLesions(frame, image_root, size=diff_size, index=index)

    assert_matched_sources(gan_ds, diff_ds)

    gan_loader = DataLoader(gan_ds, batch_size=gan_batch, shuffle=True,
                            num_workers=num_workers, drop_last=True)
    diff_loader = DataLoader(diff_ds, batch_size=diff_batch, shuffle=True,
                             num_workers=num_workers)
    return gan_loader, diff_loader, frame


def assert_matched_sources(gan_ds, diff_ds):
    """Both arms must see the same images. Differing only in resolution (D6)."""
    a = set(gan_ds.df.image_id)
    b = set(diff_ds.df.image_id)
    if a != b:
        raise AssertionError(
            f"Generative arms have different source data: "
            f"{len(a)} vs {len(b)} images, "
            f"{len(a ^ b)} not shared. The equal-budget control requires "
            f"identical source images, split and class subset; only the "
            f"resize differs (128 vs 512, per D6)."
        )
    print(f"[data] source data matched across arms: {len(a)} images, "
          f"classes {sorted(gan_ds.df.dx.unique())}")
    return True


# --- loaders ---------------------------------------------------------------

def make_loader(ds, batch_size=32, shuffle=True, balanced=False,
                num_workers=0):
    """DataLoader for classifier conditions.

    num_workers defaults to 0. On Windows, workers > 0 inside a Jupyter
    notebook will hang or spawn incorrectly. Raise it only in .py scripts
    guarded by `if __name__ == "__main__":`, or under WSL2.

    balanced=True is inverse-frequency sampling. Use it ONLY as an ablation:
    in a main condition it confounds the augmentation comparison.
    """
    if balanced:
        labels = ds.df.dx.map(ds.c2i).values
        freq = pd.Series(labels).value_counts()
        weights = [1.0 / freq[l] for l in labels]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)


def assert_no_synthetic_leakage(*csv_paths):
    """No synthetic image may enter validation or test, under any condition.

    Run before EVERY training call. Evaluating on generated images would
    measure the generator rather than the classifier.
    """
    for path in csv_paths:
        df = pd.read_csv(path)
        if "synthetic" in df.columns and df["synthetic"].fillna(False).any():
            raise AssertionError(f"SYNTHETIC ROWS IN {path}")
        if "path" in df.columns:
            paths = df["path"].dropna().astype(str)
            bad = paths[paths.str.contains("synthetic", case=False)]
            if len(bad):
                raise AssertionError(
                    f"{len(bad)} rows in {path} point into a synthetic folder"
                )
    print(f"[data] no synthetic leakage in {len(csv_paths)} partition(s)")
    return True
