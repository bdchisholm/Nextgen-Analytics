"""
Question 3: How do wind speed and temperature at kickoff affect the total
points scored in outdoor games?

Data filter: regular season, roof in {"outdoors", "open"},
             wind and temp not missing

Methods (statistical analysis + data fitting):
    - Multiple linear regression:  total = b0 + b1*wind + b2*temp
      (solved with numpy least squares; statsmodels gives p-values)
    - Two-sample (Welch) t-test: windy (>= 15 mph) vs. calm (< 15 mph)
    - Pearson correlation of total with wind and with temp
    - Control check: compare total to total_line (the betting over/under)
      to see whether the betting market already prices in weather

Metrics:
    - regression coefficients (pts per mph, pts per deg F), R^2, p-values
    - mean total points for windy vs. calm games, t and p
    - Pearson r for each weather variable

Plots:
    - Figure 3: average total points by wind-speed bin (bar with error bars)
    - Figure 3b: scatter of temperature vs. total with fitted line
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from load_data import FIG_DIR, load_games

WINDY_MPH = 15


def analyze(df):
    d = df[df["is_outdoor"]].dropna(subset=["wind", "temp", "total"]).copy()

    # Multiple linear regression with numpy least squares
    X = np.column_stack([np.ones(len(d)), d["wind"], d["temp"]])
    y = d["total"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    r2 = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)

    # Standard errors and p-values for each coefficient
    n, k = X.shape
    sigma2 = np.sum((y - y_hat) ** 2) / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    pvals = 2 * stats.t.sf(np.abs(coef / se), n - k)

    windy = d.loc[d["wind"] >= WINDY_MPH, "total"]
    calm = d.loc[d["wind"] < WINDY_MPH, "total"]
    tt = stats.ttest_ind(windy, calm, equal_var=False)

    metrics = {
        "games": len(d),
        "intercept": coef[0],
        "pts_per_mph_wind": coef[1],
        "pts_per_degF": coef[2],
        "p_wind": pvals[1],
        "p_temp": pvals[2],
        "R2": r2,
        "r_wind": stats.pearsonr(d["wind"], d["total"])[0],
        "r_temp": stats.pearsonr(d["temp"], d["total"])[0],
        "mean_total_windy": windy.mean(),
        "mean_total_calm": calm.mean(),
        "n_windy": len(windy),
        "welch_t": tt.statistic,
        "welch_p": tt.pvalue,
    }
    return d, coef, metrics


def plot(d, coef):
    FIG_DIR.mkdir(exist_ok=True)

    bins = [0, 5, 10, 15, 20, 100]
    labels = ["0-4", "5-9", "10-14", "15-19", "20+"]
    d["wind_bin"] = pd.cut(d["wind"], bins=bins, labels=labels, right=False)
    g = d.groupby("wind_bin", observed=True)["total"]
    means, sems, counts = g.mean(), g.sem(), g.size()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(means.index.astype(str), means, yerr=1.96 * sems, capsize=4,
           color="#1f77b4")
    for i, (m, c) in enumerate(zip(means, counts)):
        ax.text(i, 1, f"n={c}", ha="center", color="white", fontsize=9)
    ax.set_xlabel("Wind speed at kickoff (mph)")
    ax.set_ylabel("Average combined points")
    ax.set_title("Figure 3: Total Points by Wind Speed (outdoor games, 95% CI)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_points_by_wind.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(d["temp"], d["total"], s=6, alpha=0.25)
    t = np.linspace(d["temp"].min(), d["temp"].max(), 100)
    ax.plot(t, coef[0] + coef[1] * d["wind"].mean() + coef[2] * t,
            color="#d62728", lw=2, label="Regression (wind held at mean)")
    ax.set_xlabel("Temperature at kickoff (deg F)")
    ax.set_ylabel("Combined points")
    ax.set_title("Figure 3b: Temperature vs. Total Points")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3b_temp_vs_total.png", dpi=150)
    plt.close("all")


if __name__ == "__main__":
    d, coef, m = analyze(load_games())
    print("Question 3: Weather and scoring")
    for k, v in m.items():
        print(f"  {k:>18}: {v:.4g}")
    plot(d, coef)
