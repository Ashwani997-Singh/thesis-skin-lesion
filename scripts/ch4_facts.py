"""ch4_facts.py - prints every number Chapter 4 needs, read-only.

Run from D:\\thesis with the venv active:
    python scripts\\ch4_facts.py > results\\ch4_facts_output.txt
It only READS the metadata, the split CSVs and the image headers. It writes nothing
except what you redirect. No image data is printed.
"""
import glob, os, sys
from collections import Counter
import pandas as pd

ROOT = os.getcwd()

def find(pattern):
    hits = [p for p in glob.glob(os.path.join(ROOT, "data", "**", pattern), recursive=True)
            if ".venv" not in p]
    return sorted(hits)

meta_paths = find("HAM10000_metadata*.csv")
if not meta_paths:
    sys.exit("HAM10000_metadata.csv not found under data/ - tell Claude the path")
meta = pd.read_csv(meta_paths[0])
print("=== 1. METADATA ===")
print("file:", os.path.relpath(meta_paths[0]))
print("shape:", meta.shape)
print("columns:", list(meta.columns))
print("\nmissing values per column:")
print(meta.isna().sum().to_string())
for col in ["dx_type", "sex", "localization"]:
    if col in meta:
        print(f"\n{col} value counts:")
        print(meta[col].value_counts(dropna=False).to_string())
if "age" in meta:
    print("\nage: min %.0f  median %.0f  max %.0f  missing %d" % (
        meta.age.min(), meta.age.median(), meta.age.max(), meta.age.isna().sum()))
print("\nduplicate image_id rows:", int(meta.image_id.duplicated().sum()))

print("\n=== 2. CLASSES, IMAGES AND LESIONS ===")
n = len(meta)
per = meta.groupby("dx").agg(images=("image_id", "size"), lesions=("lesion_id", "nunique"))
per["img_pct"] = (100 * per.images / n).round(1)
ipl = meta.groupby("lesion_id").size()
lesion_dx = meta.groupby("lesion_id").dx.first()
multi = ipl[ipl > 1]
per["lesions_multi"] = lesion_dx[multi.index].value_counts().reindex(per.index).fillna(0).astype(int)
per["pct_lesions_multi"] = (100 * per.lesions_multi / per.lesions).round(1)
per["mean_img_per_lesion"] = (per.images / per.lesions).round(2)
print(per.sort_values("images", ascending=False).to_string())
print("\ntotal images:", n, " total lesions:", meta.lesion_id.nunique())
print("lesions with >1 image:", len(multi), " images in them:", int(multi.sum()),
      f"({100 * multi.sum() / n:.1f}% of images)")
print("images per lesion distribution:", dict(sorted(Counter(ipl.values).items())))
print("imbalance ratio largest/smallest class: %.1f" % (per.images.max() / per.images.min()))
print("lesions spanning >1 dx label:", int((meta.groupby('lesion_id').dx.nunique() > 1).sum()))
if "dx_type" in meta:
    print("\ndx_type by class (counts):")
    print(pd.crosstab(meta.dx, meta.dx_type).to_string())

print("\n=== 3. IMAGE FILES (headers only) ===")
img_paths = [p for p in glob.glob(os.path.join(ROOT, "data", "raw", "**", "*.jpg"), recursive=True)]
print("jpg files found:", len(img_paths))
try:
    from PIL import Image
    sizes, modes, bad = Counter(), Counter(), 0
    for p in img_paths:
        try:
            with Image.open(p) as im:
                sizes[im.size] += 1; modes[im.mode] += 1
        except Exception:
            bad += 1
    print("sizes (w,h):", dict(sizes))
    print("modes:", dict(modes), " unreadable:", bad)
    ids_files = {os.path.splitext(os.path.basename(p))[0] for p in img_paths}
    print("metadata ids without a file:", len(set(meta.image_id) - ids_files),
          " files without metadata:", len(ids_files - set(meta.image_id)))
except ImportError:
    print("PIL not installed - skip")

print("\n=== 4. SPLITS ===")
splits = {}
for name in ["train", "val", "test"]:
    hits = find(f"{name}.csv")
    if hits:
        splits[name] = pd.read_csv(hits[0])
        print(f"{name}: {os.path.relpath(hits[0])}  rows {len(splits[name])}  columns {list(splits[name].columns)}")
if len(splits) == 3:
    for name, df in splits.items():
        if "lesion_id" not in df:
            splits[name] = df.merge(meta[["image_id", "lesion_id", "dx"]], on="image_id", how="left",
                                    suffixes=("", "_m"))
    tab = pd.DataFrame({k: v.dx.value_counts() for k, v in splits.items()}).fillna(0).astype(int)
    tab["total"] = tab.sum(1)
    for k in splits:
        tab[k + "_%ofclass"] = (100 * tab[k] / tab.total).round(1)
    print(tab.to_string())
    print("\nrows per split:", {k: len(v) for k, v in splits.items()},
          " lesions per split:", {k: v.lesion_id.nunique() for k, v in splits.items()})
    L = {k: set(v.lesion_id) for k, v in splits.items()}
    I = {k: set(v.image_id) for k, v in splits.items()}
    print("CHECK lesion overlap train-val:", len(L["train"] & L["val"]),
          " train-test:", len(L["train"] & L["test"]), " val-test:", len(L["val"] & L["test"]))
    print("CHECK image overlap train-val:", len(I["train"] & I["val"]),
          " train-test:", len(I["train"] & I["test"]), " val-test:", len(I["val"] & I["test"]))
    allids = I["train"] | I["val"] | I["test"]
    print("CHECK every metadata image assigned once:", allids == set(meta.image_id),
          " total assigned:", sum(len(v) for v in I.values()))
    print("CHECK synthetic ids in val/test:",
          sum(1 for k in ["val", "test"] for i in I[k] if not str(i).startswith("ISIC_")))
    tr = splits["train"]
    src = tr[tr.dx.isin(["mel", "bcc", "akiec", "vasc", "df"])]
    print("\n=== 5. GENERATOR SOURCE SET (train, five minority classes) ===")
    g = src.groupby("dx").agg(images=("image_id", "size"), lesions=("lesion_id", "nunique"))
    print(g.to_string()); print("total images:", len(src), " lesions:", src.lesion_id.nunique())
else:
    print("split CSVs not all found:", list(splits))
