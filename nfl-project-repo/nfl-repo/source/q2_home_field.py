"""
Question 2: How large is home-field advantage in the NFL, and has it
changed over time?

Data filter: regular season, location == "Home" (neutral-site games removed)

Methods (statistical analysis):
    - One-sample t-test:  H0: mean home margin = 0  vs.  H1: > 0
    - Two-sample (Welch) t-test: 1999-2018 vs. 2019-present home margin
    - Linear regression of season-average home margin on season (trend)
    - 95% confidence interval for the mean home margin

Metrics:
    - mean home margin (pts) and 95% CI
    - home win percentage
    - per-season mean home margin
    - t statistic and p-value for each test
    - slope of the trend line (pts per season)

Plots:
    - Figure 2: average home margin per season (bar) with trend line
      and the 2020 "no fans" season highlighted
    - Figure 2b: home win % per season
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from load_data import FIG_DIR, load_games

SPLIT_SEASON = 2019  # first season of the "recent" era


def analyze(df):
    d = df[df["location"] == "Home"].copy()
    margin = d["result"]

    one = stats.ttest_1samp(margin, 0, alternative="greater")
    ci = stats.t.interval(0.95, len(margin) - 1,
                          loc=margin.mean(), scale=stats.sem(margin))

    early = d.loc[d["season"] < SPLIT_SEASON, "result"]
    recent = d.loc[d["season"] >= SPLIT_SEASON, "result"]
    two = stats.ttest_ind(early, recent, equal_var=False)

    by_season = d.groupby("season").agg(
        mean_margin=("result", "mean"),
        home_win_pct=("home_win", "mean"),
        games=("result", "size"),
    )
    by_season["home_win_pct"] *= 100
    trend = stats.linregress(by_season.index, by_season["mean_margin"])

    metrics = {
        "games": len(d),
        "mean_home_margin": margin.mean(),
        "CI95_low": ci[0],
        "CI95_high": ci[1],
        "home_win_pct": d["home_win"].mean() * 100,
        "one_sample_t": one.statistic,
        "one_sample_p": one.pvalue,
        f"mean_margin_pre{SPLIT_SEASON}": early.mean(),
        f"mean_margin_{SPLIT_SEASON}on": recent.mean(),
        "welch_t": two.statistic,
        "welch_p": two.pvalue,
        "trend_pts_per_season": trend.slope,
        "trend_p": trend.pvalue,
    }
    return by_season, trend, metrics


def plot(by_season, trend):
    FIG_DIR.mkdir(exist_ok=True)
    seasons = by_season.index.to_numpy()
    colors = ["#d62728" if s == 2020 else "#1f77b4" for s in seasons]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(seasons, by_season["mean_margin"], color=colors)
    ax.plot(seasons, trend.slope * seasons + trend.intercept, "k--",
            label=f"Trend: {trend.slope:+.3f} pts/season")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Season")
    ax.set_ylabel("Average home margin (pts)")
    ax.set_title("Figure 2: Home-Field Advantage by Season (2020 = red, few fans)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_home_field_by_season.png", dpi=150)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(seasons, by_season["home_win_pct"], marker="o")
    ax.axhline(50, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Season")
    ax.set_ylabel("Home win %")
    ax.set_title("Figure 2b: Home Win Percentage by Season")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2b_home_win_pct.png", dpi=150)
    plt.close("all")


if __name__ == "__main__":
    by_season, trend, m = analyze(load_games())
    print("Question 2: Home-field advantage")
    for k, v in m.items():
        print(f"  {k:>22}: {v:.4g}")
    plot(by_season, trend)
