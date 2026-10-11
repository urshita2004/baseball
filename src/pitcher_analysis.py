"""Descriptive pitcher analysis on retained rows of data/processed/pitches.parquet.

Writes reports/pitcher_summary.csv (all pitchers x pitch types), reports/pitcher_game_summary.csv
(per game), figures under reports/figures/, and reports/pitcher_analysis.md for one pitcher.

Usage: .venv/bin/python src/pitcher_analysis.py [--pitcher-id ID]
Without --pitcher-id the detailed report goes to the pitcher with the most pitches among those
with at least two games in the sample (so a game-to-game comparison is possible).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
LOW_PITCHES, LOW_SWINGS = 30, 50  # display heuristics, not reliability guarantees


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes in n trials; (nan, nan) when n == 0."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return np.where(n > 0, centre - half, np.nan), np.where(n > 0, centre + half, np.nan)


def load(path=ROOT / "data" / "processed" / "pitches.parquet"):
    df = pd.read_parquet(path)
    return df[df["status"] == "retained"].copy()


def summarize(df, by):
    """Tracking and outcome summary. Each rate carries its own numerator and denominator."""
    g = df.groupby(by, dropna=False)
    out = g.agg(
        pitches=("pitch_key", "size"),
        games=("game_pk", "nunique"),
        velo_mean=("velocity_mph", "mean"), velo_median=("velocity_mph", "median"), velo_sd=("velocity_mph", "std"),
        velo_n=("velocity_mph", "count"),
        spin_mean=("spin_rpm", "mean"), spin_n=("spin_rpm", "count"),
        hmov_arm_in=("hmov_arm_in", "mean"), ivb_in=("ivb_in", "mean"),
        extension_ft=("extension_ft", "mean"),
        release_x_sd_in=("release_x_ft", lambda s: s.std() * 12), release_z_sd_in=("release_z_ft", lambda s: s.std() * 12),
        swings=("eligible_swing", "sum"), misses=("is_miss", lambda s: int((s == 1).sum())),
        classified=("outcome_category", lambda s: int(s.isin(["miss", "contact", "take", "bunt"]).sum())),
        in_zone=("in_zone", lambda s: int(s.fillna(False).sum())), zone_n=("in_zone", "count"),
    )
    out["whiff_rate"] = out["misses"] / out["swings"].where(out["swings"] > 0)
    out["whiff_lo"], out["whiff_hi"] = wilson(out["misses"], out["swings"])
    out["swstr_rate"] = out["misses"] / out["classified"].where(out["classified"] > 0)
    out["zone_rate"] = out["in_zone"] / out["zone_n"].where(out["zone_n"] > 0)
    out["low_sample_tracking"] = out["pitches"] < LOW_PITCHES
    out["low_sample_swings"] = out["swings"] < LOW_SWINGS
    return out


def pitcher_tables(df):
    by_type = summarize(df, ["pitcher_id", "pitcher_name", "p_throws", "pitch_type_raw"]).reset_index()
    known = df[df["pitch_type_raw"].notna()].groupby("pitcher_id").size().rename("known_type_pitches")
    by_type = by_type.merge(known, on="pitcher_id", how="left")
    by_type["usage"] = by_type["pitches"] / by_type["known_type_pitches"]
    by_game = summarize(df, ["pitcher_id", "pitcher_name", "game_pk", "game_date", "pitch_type_raw"]).reset_index()
    overall = summarize(df, ["pitcher_id", "pitcher_name", "p_throws"]).reset_index()
    return overall, by_type, by_game


def choose_pitcher(overall):
    eligible = overall[overall["games"] >= 2]
    return int(eligible.sort_values(["pitches", "pitcher_id"], ascending=[False, True]).iloc[0]["pitcher_id"])


def peer_context(df, pid):
    """Same week, same pitch type and pitcher handedness, selected pitcher excluded."""
    me = df[df["pitcher_id"] == pid]
    hand = me["p_throws"].iloc[0]
    peers = df[(df["pitcher_id"] != pid) & (df["p_throws"] == hand)]
    p = summarize(peers, ["pitch_type_raw"])
    return p[["pitchers_placeholder"] if False else ["pitches", "velo_median", "ivb_in", "hmov_arm_in", "swings", "misses", "whiff_rate", "zone_rate"]].join(
        peers.groupby("pitch_type_raw")["pitcher_id"].nunique().rename("peer_pitchers"))


def figures(df, pid, name):
    me = df[df["pitcher_id"] == pid]
    types = me["pitch_type_raw"].value_counts().index.tolist()
    colors = dict(zip(types, plt.cm.tab10.colors))
    paths = {}

    fig, ax = plt.subplots(figsize=(6, 3.2))
    counts = me["pitch_type_raw"].value_counts()
    ax.barh(counts.index[::-1], counts.values[::-1], color=[colors[t] for t in counts.index[::-1]])
    for i, (t, v) in enumerate(zip(counts.index[::-1], counts.values[::-1])):
        ax.text(v, i, f" {v} ({v / counts.sum():.0%})", va="center", fontsize=8)
    ax.set_xlabel("pitches"); ax.set_title(f"{name}: pitch mix (n={counts.sum()})")
    ax.set_xlim(0, counts.max() * 1.25)
    paths["mix"] = save(fig, f"{pid}_pitch_mix.png")

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for t in types:
        s = me[me["pitch_type_raw"] == t]
        ax.scatter(s["hmov_arm_in"], s["ivb_in"], s=10, alpha=0.6, color=colors[t], label=f"{t} (n={s['hmov_arm_in'].notna().sum()})")
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("horizontal movement, arm-side + (in)"); ax.set_ylabel("induced vertical break (in)")
    ax.set_title(f"{name}: movement profile"); ax.legend(fontsize=8)
    paths["movement"] = save(fig, f"{pid}_movement.png")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for t in types:
        s = me[me["pitch_type_raw"] == t]
        ax.scatter(s["release_x_arm_ft"], s["release_z_ft"], s=10, alpha=0.6, color=colors[t], label=t)
    ax.set_xlabel("release side, arm-side + (ft)"); ax.set_ylabel("release height (ft)")
    ax.set_title(f"{name}: release points"); ax.legend(fontsize=8)
    paths["release"] = save(fig, f"{pid}_release.png")

    games = sorted(me["game_date"].unique())
    fig, axes = plt.subplots(1, len(types), figsize=(2.4 * len(types), 3.4), sharey=False, squeeze=False)
    for ax, t in zip(axes[0], types):
        data = [me[(me["pitch_type_raw"] == t) & (me["game_date"] == g)]["velocity_mph"].dropna() for g in games]
        ax.boxplot(data, tick_labels=[f"{pd.Timestamp(g):%m-%d}\nn={len(d)}" for g, d in zip(games, data)])
        ax.set_title(t, fontsize=9); ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel("velocity (mph)")
    fig.suptitle(f"{name}: velocity by game", fontsize=10)
    paths["velo"] = save(fig, f"{pid}_velocity_by_game.png")

    # Location: scatter plus pitch counts in a 3x3 grid around the zone (counts before any heatmap reading).
    fig, axes = plt.subplots(1, len(types), figsize=(2.6 * len(types), 3.4), squeeze=False)
    xe = np.array([-np.inf, -0.83, -0.28, 0.28, 0.83, np.inf])
    for ax, t in zip(axes[0], types):
        s = me[(me["pitch_type_raw"] == t)].dropna(subset=["plate_x_ft", "plate_z_ft"])
        ax.scatter(s["plate_x_ft"], s["plate_z_ft"], s=6, alpha=0.5, color=colors[t])
        top, bot = me["sz_top_ft"].median(), me["sz_bot_ft"].median()
        ax.add_patch(plt.Rectangle((-0.83, bot), 1.66, top - bot, fill=False, lw=1))
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-0.5, 5); ax.set_aspect("equal")
        ax.set_title(f"{t} (n={len(s)}, zone {s['in_zone'].mean():.0%})", fontsize=8); ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel("plate height (ft)")
    fig.suptitle(f"{name}: plate location, catcher's view (median zone shown)", fontsize=10)
    paths["location"] = save(fig, f"{pid}_location.png")
    return paths


def save(fig, fname):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=130)
    plt.close(fig)
    return f"figures/{fname}"


def fmt(x, nd=1, pct=False):
    if pd.isna(x):
        return "—"
    return f"{x:.{nd}%}" if pct else f"{x:.{nd}f}"


def report(df, overall, by_type, by_game, pid, paths):
    me_o = overall[overall["pitcher_id"] == pid].iloc[0]
    name, hand = me_o["pitcher_name"], me_o["p_throws"]
    t = by_type[by_type["pitcher_id"] == pid].sort_values("pitches", ascending=False)
    g = by_game[by_game["pitcher_id"] == pid]
    peers = peer_context(df, pid)
    me = df[df["pitcher_id"] == pid]

    rep = pd.DataFrame({
        "pitch": t["pitch_type_raw"], "n": t["pitches"], "usage": t["usage"].map(lambda v: fmt(v, 1, True)),
        "velo mph (mean ± sd)": [f"{m:.1f} ± {s:.1f}" for m, s in zip(t["velo_mean"], t["velo_sd"])],
        "spin rpm": t["spin_mean"].map(lambda v: fmt(v, 0)),
        "IVB in": t["ivb_in"].map(fmt), "arm-side HB in": t["hmov_arm_in"].map(fmt),
        "ext ft": t["extension_ft"].map(lambda v: fmt(v, 2)),
        "release sd x/z in": [f"{a:.1f} / {b:.1f}" for a, b in zip(t["release_x_sd_in"], t["release_z_sd_in"])],
    })
    out = pd.DataFrame({
        "pitch": t["pitch_type_raw"],
        "whiffs / swings": [f"{m} / {s}" for m, s in zip(t["misses"], t["swings"])],
        "whiff rate (95% CI)": [f"{fmt(r, 1, True)} ({fmt(lo, 0, True)}–{fmt(hi, 0, True)})" for r, lo, hi in zip(t["whiff_rate"], t["whiff_lo"], t["whiff_hi"])],
        "SwStr (misses / classified)": [f"{fmt(r, 1, True)} ({m}/{c})" for r, m, c in zip(t["swstr_rate"], t["misses"], t["classified"])],
        "zone rate (in / located)": [f"{fmt(r, 1, True)} ({i}/{n})" for r, i, n in zip(t["zone_rate"], t["in_zone"], t["zone_n"])],
        "sample": ["low (swings < 50)" if low else "ok" for low in t["low_sample_swings"]],
    })
    pc = peers.loc[peers.index.intersection(t["pitch_type_raw"])].reindex(t["pitch_type_raw"])
    ctx = pd.DataFrame({
        "pitch": t["pitch_type_raw"].values,
        "player velo": t["velo_median"].map(fmt).values, f"peer {hand}HP velo": pc["velo_median"].map(fmt).values,
        "player IVB": t["ivb_in"].map(fmt).values, "peer IVB": pc["ivb_in"].map(fmt).values,
        "player HB": t["hmov_arm_in"].map(fmt).values, "peer HB": pc["hmov_arm_in"].map(fmt).values,
        "player whiff": t["whiff_rate"].map(lambda v: fmt(v, 1, True)).values,
        "peer whiff (misses/swings)": [f"{fmt(r, 1, True)} ({int(m)}/{int(s)})" if pd.notna(s) else "—" for r, m, s in zip(pc["whiff_rate"], pc["misses"].fillna(0), pc["swings"])],
        "peer pitchers": pc["peer_pitchers"].fillna(0).astype(int).values,
    })
    primary = t.iloc[0]["pitch_type_raw"]
    games = (me.groupby(["game_date", "game_pk"])
               .agg(pitches=("pitch_key", "size"), swings=("eligible_swing", "sum"),
                    misses=("is_miss", lambda s: int((s == 1).sum())))
               .join(me[me["pitch_type_raw"] == primary].groupby(["game_date", "game_pk"])["velocity_mph"]
                     .agg(["mean", "count"]).rename(columns={"mean": f"{primary} velo mean", "count": f"{primary} n"}))
               .reset_index().sort_values("game_date"))
    pg = g[g["pitch_type_raw"] == primary].sort_values("game_date")[["game_date", "pitches", "velo_mean", "velo_sd", "release_x_sd_in", "release_z_sd_in", "swings", "misses"]]

    vs = me.groupby("stand").agg(pitches=("pitch_key", "size"), swings=("eligible_swing", "sum"),
                                 misses=("is_miss", lambda s: int((s == 1).sum())))
    vs["whiff"] = (vs["misses"] / vs["swings"]).map(lambda v: fmt(v, 1, True))

    L = [f"# Pitcher analysis — {name} ({hand}HP, MLBAM {pid})", "",
         f"Sample: Statcast regular season 2025-06-01 to 2025-06-07, retained rows only. {int(me_o['pitches'])} pitches "
         f"in {int(me_o['games'])} games. Generated by `src/pitcher_analysis.py`; every number traces to "
         f"`data/processed/pitches.parquet` rows with `pitcher_id == {pid}`.", "",
         f"Selection rule: most pitches among pitchers with ≥ 2 games in the sample (to allow a game-to-game view). "
         "This is a coverage choice, not a quality ranking.", "",
         "**Units & conventions.** Velocity mph; spin rpm; movement in inches (IVB = induced vertical break, gravity removed; "
         "HB = horizontal break, arm-side positive); release in ft, release spread as standard deviation in inches; plate location "
         "in ft from the catcher's view. Zone = rulebook plate width ± ball radius, height from each pitch's `sz_top`/`sz_bot`.", "",
         "**Denominators.** Usage = pitches of type / pitches with known type. Whiff rate = misses / ordinary (non-bunt) swings. "
         "SwStr = misses / pitches with a classified outcome (a different statistic). Zone rate = in-zone / pitches with location.", "",
         "## Repertoire and pitch characteristics", "", rep.to_markdown(index=False), "",
         f"![pitch mix]({paths['mix']})", "", f"![movement]({paths['movement']})", "",
         "## Release consistency", "",
         f"Spread of release position within pitch type (whole sample) is in the table above. Per game for the primary pitch ({primary}):", "",
         pg.assign(game_date=pg["game_date"].astype(str)).round(2).to_markdown(index=False), "",
         f"![release]({paths['release']})", "",
         "## Velocity by game", "",
         games.assign(game_date=games["game_date"].astype(str)).round(1).to_markdown(index=False), "",
         f"![velocity]({paths['velo']})", "",
         "With only a handful of games, differences between games are descriptive; they do not establish a trend.", "",
         "## Location", "",
         "Without catcher targets this describes where pitches ended up, not command accuracy.", "",
         f"![location]({paths['location']})", "",
         "## Observed swing outcomes", "", out.to_markdown(index=False), "",
         "By batter side (all pitch types):", "", vs.reset_index().to_markdown(index=False), "",
         "## Context: same-week peers", "",
         f"Medians/means over other {hand}HP pitchers' pitches of the same type in the same week (this pitcher excluded). "
         "Peer whiff rates pool pitches, so high-volume pitchers weigh more; batters faced differ. One week is a small window.", "",
         ctx.to_markdown(index=False), "",
         "## Observations vs hypotheses", "",
         "**Observed** (measured in this sample): the tables and figures above. Each rate shows its numerator and denominator.", "",
         "**Hypotheses** (not tested here; candidates for later modelling):", "",
         *hypotheses(t, ctx, hand), "",
         "None of these establish that changing a pitch characteristic would change outcomes; that requires "
         "a practice-experiment design.", ""]
    (ROOT / "reports" / "pitcher_analysis.md").write_text("\n".join(L))


def hypotheses(t, ctx, hand):
    """Neutral prompts generated from the data, worded as questions."""
    hs = []
    for _, r in t.iterrows():
        if r["swings"] >= LOW_SWINGS and pd.notna(r["whiff_rate"]):
            hs.append(f"- {r['pitch_type_raw']}: whiff rate {r['whiff_rate']:.0%} on {int(r['swings'])} swings "
                      f"(95% CI {r['whiff_lo']:.0%}–{r['whiff_hi']:.0%}). Does this hold over a full season of swings?")
        elif r["swings"] > 0:
            hs.append(f"- {r['pitch_type_raw']}: {int(r['misses'])}/{int(r['swings'])} whiffs, below the {LOW_SWINGS}-swing display "
                      f"threshold (95% CI {r['whiff_lo']:.0%}–{r['whiff_hi']:.0%}). Too wide to compare with peers yet.")
    big = t.sort_values("release_z_sd_in", ascending=False).iloc[0]
    hs.append(f"- Largest release-height spread is on the {big['pitch_type_raw']} ({big['release_z_sd_in']:.1f} in sd, "
              f"n={int(big['pitches'])}). Is that spread associated with movement or location differences within the pitch type?")
    return hs


def main(pitcher_id=None):
    df = load()
    overall, by_type, by_game = pitcher_tables(df)
    (ROOT / "reports").mkdir(exist_ok=True)
    by_type.round(4).to_csv(ROOT / "reports" / "pitcher_summary.csv", index=False)
    by_game.round(4).to_csv(ROOT / "reports" / "pitcher_game_summary.csv", index=False)
    pid = pitcher_id or choose_pitcher(overall)
    name = overall.loc[overall["pitcher_id"] == pid, "pitcher_name"].iloc[0]
    paths = figures(df, pid, name)
    report(df, overall, by_type, by_game, pid, paths)
    print(f"report for {name} ({pid}); summary rows: {len(by_type)} pitcher x pitch type")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitcher-id", type=int)
    main(**vars(ap.parse_args()))
