"""
audit_hubness.py - EXPLORATORY analysis of nearest-neighbour concentration.

Not declared in D22/D24/D25; reported as exploratory. Uses only the per-image
audit outputs already produced, so no new distance is computed and no declared
threshold or rule is changed.

Question: do many synthetic images share one nearest real image (a "hub"), and
is that hub also a hub among real images? If real images do not choose it,
the concentration belongs to the generator's outputs, not to the space itself.

Usage (from D:\\thesis):  python audit_hubness.py
Writes results/run1_cloud/metrics/audit_hubness.csv
"""

from pathlib import Path

import pandas as pd

M = Path("results/run1_cloud/metrics")
h = pd.read_csv(M / "audit_headline_per_image.csv")
r = pd.read_csv(M / "audit_real_to_real.csv")
real_hub_max = int(r.nn_image_id.value_counts().iloc[0])
rows = []
for arm, g in h.groupby("arm"):
    vc = g.nn_image_id.value_counts()
    hub = vc.index[0]
    hub_dx = r.loc[r.image_id == hub, "dx"].iloc[0]
    row = {"arm": arm, "distinct_real_nn": int(g.nn_image_id.nunique()),
           "hub_image": hub, "hub_class": hub_dx,
           "hub_count": int(vc.iloc[0]), "hub_share": round(vc.iloc[0] / len(g), 4),
           "real_images_with_hub_as_nn": int((r.nn_image_id == hub).sum()),
           "max_real_images_sharing_one_real_nn": real_hub_max,
           "same_class_nn_share": round(float((g.dx == g.nn_dx).mean()), 4),
           "top10_closest_distinct_real": int(g.nsmallest(10, "nn_dist").nn_image_id.nunique()),
           "hub_min_dist": round(float(g.loc[g.nn_image_id == hub, "nn_dist"].min()), 4)}
    for c, n in g[g.nn_image_id == hub].dx.value_counts().items():
        row[f"to_hub_{c}"] = int(n)
    rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(M / "audit_hubness.csv", index=False)
print(out.T.to_string())
print(f"\nsaved {M / 'audit_hubness.csv'}")
