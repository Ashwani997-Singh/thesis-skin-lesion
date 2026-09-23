import pandas as pd
from sklearn.model_selection import train_test_split


def lesion_level_split(df, train=0.70, val=0.15, test=0.15, seed=42):
    """Split by lesion, so every image of a lesion lands in one partition."""
    assert abs(train + val + test - 1.0) < 1e-9, "fractions must sum to 1"

    # Safety check: every image of one lesion must share one diagnosis,
    # otherwise taking "first" below would be wrong.
    assert df.groupby("lesion_id").dx.nunique().max() == 1, \
        "a lesion has more than one diagnosis"

    # 1. One row per lesion
    lesions = df.groupby("lesion_id").agg(dx=("dx", "first")).reset_index()

    # 2. Split lesions 70 / 30, keeping class proportions
    tr_l, tmp_l = train_test_split(
        lesions, train_size=train, stratify=lesions.dx, random_state=seed)

    # 3. Split the 30 into 15 / 15
    va_l, te_l = train_test_split(
        tmp_l, train_size=val / (val + test), stratify=tmp_l.dx,
        random_state=seed)

    # 4. Bring ALL images back, following their lesion
    return {
        "train": df[df.lesion_id.isin(tr_l.lesion_id)].copy(),
        "val":   df[df.lesion_id.isin(va_l.lesion_id)].copy(),
        "test":  df[df.lesion_id.isin(te_l.lesion_id)].copy(),
    }


def check_split(splits, df):
    """The four leakage assertions. Must all pass."""
    tr, va, te = splits["train"], splits["val"], splits["test"]
    a, b, c = set(tr.lesion_id), set(va.lesion_id), set(te.lesion_id)

    assert not (a & c), "LEAK: lesion in both train and test"
    assert not (a & b), "LEAK: lesion in both train and val"
    assert not (b & c), "LEAK: lesion in both val and test"
    assert len(tr) + len(va) + len(te) == len(df), "images lost"
    print("All four leakage checks passed.")