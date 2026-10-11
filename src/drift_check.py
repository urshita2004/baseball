"""Label-free covariate and definition check between a reference season and a target season.

Uses model inputs only, never outcomes, so it can run on a sealed season before any scoring.
  - Distribution shift per numeric feature: population stability index (PSI) over reference deciles,
    standardized median shift.
  - Definition checks on fields that derived features depend on: within-batter variability of the strike-zone
    fields (a per-pitch, stance-based zone varies; a fixed height-based zone does not).
A feature is flagged if PSI >= 0.10, or if a definition check changes by more than 5x.

Usage: .venv/bin/python src/drift_check.py --ref statcast_2025 --target statcast_2026 [--features v1|v2]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PSI_FLAG, DEFINITION_RATIO_FLAG = 0.10, 5.0


def psi(ref, target, bins=10):
    ref, target = pd.Series(ref).dropna(), pd.Series(target).dropna()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    t = np.histogram(target, edges)[0] / len(target)
    r, t = np.clip(r, 1e-6, None), np.clip(t, 1e-6, None)
    return float(np.sum((t - r) * np.log(t / r)))


def drift_table(ref, target, numeric):
    rows = []
    for f in numeric:
        a, b = ref[f].astype(float), target[f].astype(float)
        iqr = a.quantile(0.75) - a.quantile(0.25)
        rows.append({"feature": f, "ref_median": a.median(), "target_median": b.median(),
                     "median_shift_per_iqr": (b.median() - a.median()) / iqr if iqr else np.nan,
                     "psi": psi(a, b), "ref_missing": a.isna().mean(), "target_missing": b.isna().mean()})
    out = pd.DataFrame(rows)
    out["flag"] = out["psi"] >= PSI_FLAG
    return out


def definition_checks(ref, target):
    """Within-batter variability of zone fields: median over batters of the SD of each field."""
    rows = []
    for f in ("sz_top_ft", "sz_bot_ft"):
        a = ref.groupby("batter_id")[f].std().median()
        b = target.groupby("batter_id")[f].std().median()
        ratio = (b + 1e-4) / (a + 1e-4)
        rows.append({"check": f"within-batter SD of {f}", "ref": a, "target": b, "ratio": ratio,
                     "flag": ratio > DEFINITION_RATIO_FLAG or ratio < 1 / DEFINITION_RATIO_FLAG,
                     "affects": "plate_z_zone"})
    return pd.DataFrame(rows)


def gate(ref, target, numeric):
    """(drift table, definition table, flagged features). Features depending on a flagged definition are flagged."""
    d, c = drift_table(ref, target, numeric), definition_checks(ref, target)
    flagged = set(d.loc[d["flag"], "feature"])
    for _, row in c[c["flag"]].iterrows():
        flagged |= {f for f in row["affects"].split(",") if f in numeric}
    return d, c, sorted(flagged)


def load_inputs(source):
    ds = yaml.safe_load((ROOT / "config" / "data_sources.yaml").read_text())[source]
    df = pd.read_parquet(ROOT / ds["processed_path"])
    return df[df["status"] == "retained"]


def main(ref="statcast_2025", target="statcast_2026", features="v1"):
    cfg = yaml.safe_load((ROOT / "config" / ("features.yaml" if features == "v1" else "features_v2.yaml")).read_text())
    numeric = [f for f in cfg["allow"]["numeric"]]
    a, b = load_inputs(ref), load_inputs(target)
    if features == "v2":
        from features import add_batter_height_zone
        a, b = add_batter_height_zone(a), add_batter_height_zone(b)
    d, c, flagged = gate(a, b, numeric)
    L = [f"# Label-free drift check — {ref} → {target} (features {features})", "",
         "Model inputs only; outcomes are not read. Flag rules: PSI ≥ 0.10, or a definition check changing by more than 5×.", "",
         "## Definition checks", "", c.round(4).to_markdown(index=False), "",
         "## Distribution shift", "", d.round(4).to_markdown(index=False), "",
         f"**Flagged features: {flagged or 'none'}.**", ""]
    out = ROOT / "reports" / f"drift_{ref}_to_{target}_{features}.md"
    out.write_text("\n".join(L))
    print(c.round(4).to_string()); print(d[["feature", "psi", "median_shift_per_iqr", "flag"]].round(4).to_string())
    print("FLAGGED:", flagged, "| wrote", out.relative_to(ROOT))
    return flagged


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="statcast_2025")
    ap.add_argument("--target", default="statcast_2026")
    ap.add_argument("--features", default="v1", choices=["v1", "v2"])
    main(**vars(ap.parse_args()))
