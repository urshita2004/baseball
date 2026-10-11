"""Model features from config/features.yaml (allowlist v1).

Row-level features come straight from the processed table. Pitcher-relative fastball features are
computed as-of: each pitch is compared with the pitcher's primary fastball using only pitches thrown
strictly before it (earlier games, and earlier pitches in the same game). They use tracking data only,
never outcomes, and are computed on the full pitch table (takes included) before rows are filtered to swings.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
FEATURES = yaml.safe_load((ROOT / "config" / "features.yaml").read_text())
NUMERIC = list(FEATURES["allow"]["numeric"])
CATEGORICAL = list(FEATURES["allow"]["categorical"])
FB_DERIVED = ["velo_diff_fb", "ivb_diff_fb", "hmov_diff_fb"]
FB_TYPES = ["FF", "SI"]          # primary fastball = the more-thrown of these so far; FC only if neither thrown
FB_FALLBACK = "FC"
MIN_FB_PITCHES = 10              # fewer prior fastballs than this -> reference missing (too noisy)
REF_COLS = {"velocity_mph": "velo", "ivb_in": "ivb", "hmov_arm_in": "hmov"}
CHRONO = ["pitcher_id", "game_date", "game_pk", "at_bat_number", "pitch_number"]


def fastball_reference(pitches: pd.DataFrame) -> pd.DataFrame:
    """As-of primary-fastball reference per pitch. Returns columns indexed like `pitches`."""
    df = pitches[CHRONO + ["pitch_type_raw", *REF_COLS]].sort_values(CHRONO)
    counts, means = {}, {}
    for t in FB_TYPES + [FB_FALLBACK]:
        is_t = df["pitch_type_raw"].eq(t).fillna(False).astype("int64")
        counts[t] = is_t.groupby(df["pitcher_id"]).cumsum() - is_t  # strictly earlier pitches
        means[t] = {}
        for col, short in REF_COLS.items():
            val = df[col].astype("float64").where(is_t.astype(bool))
            s = val.fillna(0).groupby(df["pitcher_id"]).cumsum() - val.fillna(0)
            has = val.notna().astype("int64")
            n = has.groupby(df["pitcher_id"]).cumsum() - has
            means[t][short] = s / n.where(n > 0)
    counts = {t: c.to_numpy() for t, c in counts.items()}
    means = {t: {k: v.to_numpy() for k, v in m.items()} for t, m in means.items()}
    primary = np.where(counts["FF"] >= counts["SI"], "FF", "SI")
    n_primary = np.where(primary == "FF", counts["FF"], counts["SI"])
    use_fc = (n_primary == 0) & (counts[FB_FALLBACK] > 0)
    primary = np.where(use_fc, FB_FALLBACK, primary)
    n_primary = np.where(use_fc, counts[FB_FALLBACK], n_primary)
    out = pd.DataFrame(index=df.index)
    out["fb_ref_type"] = primary
    out["fb_ref_n"] = n_primary
    for short in REF_COLS.values():
        ref = np.select([primary == t for t in FB_TYPES + [FB_FALLBACK]],
                        [means[t][short] for t in FB_TYPES + [FB_FALLBACK]], default=np.nan)
        out[f"fb_ref_{short}"] = np.where(n_primary >= MIN_FB_PITCHES, ref, np.nan)
    out["fb_ref_missing"] = out["fb_ref_velo"].isna().astype(int)
    return out.reindex(pitches.index)


ABS_TOP, ABS_BOTTOM = 0.535, 0.27   # ABS zone as a share of batter height (MLB 2026 rule)


def add_batter_height_zone(df: pd.DataFrame, cache_name="batters") -> pd.DataFrame:
    """plate_z_abs_zone: vertical location relative to a zone computed from listed batter height the same way in
    every season (0 = bottom, 1 = top). Replaces plate_z_zone, whose sz_top/sz_bot changed definition in 2026."""
    from players import batter_heights
    heights, _ = batter_heights(df["batter_id"].dropna().unique(), cache_name)
    h = df["batter_id"].map(heights).astype(float)
    out = df.copy()
    out["batter_height_ft"] = h
    out["plate_z_abs_zone"] = (out["plate_z_ft"] - ABS_BOTTOM * h) / ((ABS_TOP - ABS_BOTTOM) * h)
    return out


def add_features(pitches: pd.DataFrame) -> pd.DataFrame:
    df = pitches.join(fastball_reference(pitches))
    df["velo_diff_fb"] = df["velocity_mph"] - df["fb_ref_velo"]
    df["ivb_diff_fb"] = df["ivb_in"] - df["fb_ref_ivb"]
    df["hmov_diff_fb"] = df["hmov_arm_in"] - df["fb_ref_hmov"]
    return df


def pitch_type_levels(train: pd.DataFrame, rare: list) -> dict:
    """Map raw codes to model levels: rare codes pooled to their group; known levels from training only."""
    levels = {t: (f"group:{grp}" if t in rare else t)
              for t, grp in train.groupby("pitch_type_raw")["pitch_group"].first().items()}
    return levels


def apply_pitch_type_levels(df: pd.DataFrame, levels: dict) -> pd.Series:
    """Unseen codes fall back to their pitch group, else 'unknown'."""
    return df["pitch_type_raw"].map(levels).fillna("group:" + df["pitch_group"].astype(str))


def model_columns():
    return NUMERIC + FB_DERIVED + ["fb_ref_missing"], CATEGORICAL
