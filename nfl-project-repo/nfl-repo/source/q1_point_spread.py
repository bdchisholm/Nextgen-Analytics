"""
Question 1: How accurately does the pre-game point spread predict the
final margin of victory?

Method: least-squares linear fit (data fitting)
    result = m * spread_line + b

Metrics:
    - slope m and intercept b  (m ~ 1, b ~ 0 means the spread is unbiased)
    - correlation coefficient R and R^2
    - mean absolute error (MAE) and root-mean-square error (RMSE) of the spread
    - % of games where the favorite won outright
    - % of games decided within 3 / 7 points of the spread

Plots:
    - Figure 1: scatter of spread vs. final margin with the fitted line
    - Figure 1b: histogram of spread error (result - spread_line)
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from load_data import FIG_DIR, load_games


def analyze(df):
    d = df.dropna(subset=["spread_line", "result"])
    x, y = d["spread_line"].to_numpy(), d["result"].to_numpy()

    fit = stats.linregress(x, y)
    err = y - x
    non_pick = x != 0
    favorite_won = np.sign(y[non_pick]) == np.sign(x[non_pick])

    metrics = {
        "games": len(d),
        "slope": fit.slope,
        "intercept": fit.intercept,
        "R": fit.rvalue,
        "R2": fit.rvalue ** 2,
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "favorite_win_pct": favorite_won.mean() * 100,
        "within_3_pct": (np.abs(err) <= 3).mean() * 100,
        "within_7_pct": (np.abs(err) <= 7).mean() * 100,
    }
    return d, fit, metrics


def plot(d, fit):
    FIG_DIR.mkdir(exist_ok=True)
    x, y = d["spread_line"], d["result"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=6, alpha=0.25, color="#1f77b4", label="Games")
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, fit.slope * xs + fit.intercept, color="#d62728", lw=2,
            label=f"Fit: y = {fit.slope:.2f}x + {fit.intercept:.2f}")
    ax.plot(xs, xs, "k--", lw=1, label="Perfect prediction (y = x)")
    ax.set_xlabel("Pre-game point spread (home team favored by, pts)")
    ax.set_ylabel("Final margin (home - away, pts)")
    ax.set_title("Figure 1: Point Spread vs. Final Margin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_spread_vs_margin.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d["result"] - d["spread_line"], bins=60, color="#1f77b4")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Spread error (final margin - spread, pts)")
    ax.set_ylabel("Number of games")
    ax.set_title("Figure 1b: Distribution of Point Spread Error")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1b_spread_error_hist.png", dpi=150)
    plt.close("all")


if __name__ == "__main__":
    d, fit, m = analyze(load_games())
    print("Question 1: Point spread accuracy")
    for k, v in m.items():
        print(f"  {k:>18}: {v:.3f}" if isinstance(v, float) else f"  {k:>18}: {v}")
    plot(d, fit)
