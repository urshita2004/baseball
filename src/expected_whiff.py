"""Pitcher-disjoint expected whiff, batter context and global feature importance.

Expected whiff for each swing comes from a model that never saw that pitcher (5-fold cross-fitting with folds
grouped by pitcher, over 2025 + 2026 swings). Inputs: v2 features, season, and an as-of batter context (the batter's
shrunken whiff-per-swing rate from earlier games only). A pitcher's observed minus expected whiff therefore compares
them with other pitchers' results on similar pitches to similar batters. It is not a skill estimate or a cause.

Writes data/pitcher_comparison/expected.parquet and data/pitcher_comparison/permutation_importance.csv.
Usage: .venv/bin/python src/expected_whiff.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from model_v2 import CAT, NUM_V2, SEED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "pitcher_comparison"
SHRINK_SWINGS = 100   # prior weight (in swings) pulling a batter's rate toward the league rate


def batter_context(sw, league_rate):
    """As-of batter whiff-per-swing from strictly earlier games, shrunk toward the league rate."""
    g = (sw.groupby(["batter_id", "game_date", "game_pk"]).agg(s=("y", "size"), m=("y", "sum"))
           .reset_index().sort_values(["batter_id", "game_date", "game_pk"]))
    g["prior_s"] = g.groupby("batter_id")["s"].cumsum() - g["s"]
    g["prior_m"] = g.groupby("batter_id")["m"].cumsum() - g["m"]
    g["batter_whiff_asof"] = (g["prior_m"] + SHRINK_SWINGS * league_rate) / (g["prior_s"] + SHRINK_SWINGS)
    g["batter_prior_swings"] = g["prior_s"]
    return sw.merge(g[["batter_id", "game_pk", "batter_whiff_asof", "batter_prior_swings"]], on=["batter_id", "game_pk"], how="left")


def build(num, cat):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    pre = ColumnTransformer([("num", "passthrough", num), ("cat", enc, cat)], verbose_feature_names_out=False)
    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, max_iter=800,
                                         early_stopping=False, categorical_features=[False] * len(num) + [True] * len(cat),
                                         random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def main():
    sw = pd.read_parquet(ROOT / "data" / "model" / "v2" / "swings_v2.parquet")
    sw["season"] = pd.to_datetime(sw["game_date"]).dt.year.astype(str)
    league = sw.loc[sw["season"] == "2025", "y"].mean()
    sw = batter_context(sw, league)
    num, cat = NUM_V2 + ["batter_whiff_asof"], CAT + ["season"]
    sw["expected"] = np.nan
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(sw, groups=sw["pitcher_id"])):
        model = build(num, cat).fit(sw.iloc[tr][num + cat], sw.iloc[tr]["y"].to_numpy())
        sw.iloc[te, sw.columns.get_loc("expected")] = model.predict_proba(sw.iloc[te][num + cat])[:, 1]
        print(f"fold {k + 1}/5 done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["pitch_key", "game_pk", "game_date", "season", "pitcher_id", "pitcher_name", "batter_id", "pitch_type_raw",
            "pitch_group", "p_throws", "stand", "y", "expected", "batter_whiff_asof", "batter_prior_swings"]
    sw[keep].to_parquet(OUT / "expected.parquet", index=False)

    # Global permutation importance for the v2 development model on August 2026 validation (sampled)
    dev_model = joblib.load(ROOT / "data" / "model" / "v2" / "model_train_only_v2.joblib")
    val = sw[sw["partition"] == "validation"].sample(30000, random_state=SEED)
    imp = permutation_importance(dev_model, val[NUM_V2 + CAT], val["y"].to_numpy(), scoring="neg_log_loss",
                                 n_repeats=5, random_state=SEED, n_jobs=-1)
    pd.DataFrame({"feature": NUM_V2 + CAT, "log_loss_increase": imp.importances_mean, "sd": imp.importances_std}) \
      .sort_values("log_loss_increase", ascending=False).to_csv(OUT / "permutation_importance.csv", index=False)
    by_season = sw.groupby("season").apply(lambda d: pd.Series({"observed": d["y"].mean(), "expected": d["expected"].mean()}), include_groups=False)
    print(by_season.round(4).to_string())


if __name__ == "__main__":
    main()
