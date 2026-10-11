"""Probability metrics, calibration summaries and two-way (game x pitcher) bootstrap comparisons."""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def calibration_summary(y, p):
    """Calibration-in-the-large (intercept with slope fixed at 1) and calibration slope, both on the logit scale."""
    z = logit(np.clip(p, EPS, 1 - EPS))
    citl = brentq(lambda a: expit(a + z).mean() - y.mean(), -5, 5)
    if np.ptp(z) == 0:  # constant predictions: slope undefined
        return citl, np.nan
    slope = LogisticRegression(C=np.inf).fit(z.reshape(-1, 1), y).coef_[0, 0]
    return citl, slope


def metrics(y, p):
    y, p = np.asarray(y, int), np.clip(np.asarray(p, float), EPS, 1 - EPS)
    citl, slope = calibration_summary(y, p)
    return {"n": len(y), "prevalence": y.mean(), "mean_pred": p.mean(), "log_loss": log_loss(y, p),
            "brier": brier_score_loss(y, p), "roc_auc": roc_auc_score(y, p),
            "avg_precision": average_precision_score(y, p), "calib_intercept": citl, "calib_slope": slope}


def two_way_bootstrap(y, preds: dict, games, pitchers, reference, reps=300, seed=20250601):
    """Differences (model - reference) in log loss, Brier and AUC with 95% intervals.

    Each replicate draws Poisson(1) weights for every game and every pitcher independently and weights each
    pitch by their product, so pitches sharing a game or a pitcher move together.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int)
    g_codes, g_n = pd.factorize(games)[0], pd.Series(games).nunique()
    p_codes, p_n = pd.factorize(pitchers)[0], pd.Series(pitchers).nunique()
    clipped = {k: np.clip(np.asarray(v, float), EPS, 1 - EPS) for k, v in preds.items()}
    ll = {k: -(y * np.log(v) + (1 - y) * np.log(1 - v)) for k, v in clipped.items()}
    br = {k: (v - y) ** 2 for k, v in clipped.items()}
    draws = {k: {"log_loss": [], "brier": [], "roc_auc": []} for k in preds if k != reference}
    for _ in range(reps):
        w = rng.poisson(1.0, g_n)[g_codes] * rng.poisson(1.0, p_n)[p_codes]
        if w.sum() == 0:
            continue
        ref_auc = roc_auc_score(y, clipped[reference], sample_weight=w)
        for k in draws:
            draws[k]["log_loss"].append(np.average(ll[k] - ll[reference], weights=w))
            draws[k]["brier"].append(np.average(br[k] - br[reference], weights=w))
            draws[k]["roc_auc"].append(roc_auc_score(y, clipped[k], sample_weight=w) - ref_auc)
    rows = []
    for k, d in draws.items():
        for m, vals in d.items():
            point = (np.mean(ll[k] - ll[reference]) if m == "log_loss" else np.mean(br[k] - br[reference]) if m == "brier"
                     else roc_auc_score(y, clipped[k]) - roc_auc_score(y, clipped[reference]))
            lo, hi = np.percentile(vals, [2.5, 97.5])
            rows.append({"model": k, "vs": reference, "metric": m, "difference": point, "lo95": lo, "hi95": hi})
    return pd.DataFrame(rows)


def reliability_table(y, p, bins=10):
    """Observed vs mean predicted rate by predicted-probability decile."""
    df = pd.DataFrame({"y": np.asarray(y, int), "p": np.asarray(p, float)})
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    return df.groupby("bin", observed=True).agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")).reset_index()
