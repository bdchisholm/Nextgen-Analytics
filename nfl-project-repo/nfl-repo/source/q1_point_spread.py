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
        "seasons": f"{d['season'].min()}-{d['season'].max()}",
        "slope": fit.slope,
        "intercept": fit.intercept,
        "R": fit.rvalue,
        "R2": fit.rvalue ** 2,
        "slope_p": fit.pvalue,
        "mean_error": err.mean(),
        "MAE": np.mean(np.abs(err)),
        "median_abs_error": np.median(np.abs(err)),
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
            label=f"Best fit: y = {fit.slope:.2f}x {fit.intercept:+.2f}  (R² = {fit.rvalue**2:.2f})")
    ax.plot(xs, xs, "k--", lw=1, label="Perfect prediction (y = x)")
    ax.set_xlabel("Pre-game point spread (home team favored by, pts)")
    ax.set_ylabel("Final margin (home - away, pts)")
    ax.set_title("Figure 1: Point Spread vs. Final Margin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_spread_vs_margin.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    err = d["result"] - d["spread_line"]
    ax.hist(err, bins=np.arange(-50.5, 51.5, 2), color="#1f77b4")
    ax.axvline(0, color="k", lw=1)
    mae = np.abs(err).mean()
    for v in (-mae, mae):
        ax.axvline(v, color="#d62728", ls="--", lw=1)
    ax.text(mae + 1, ax.get_ylim()[1] * 0.9, f"MAE = {mae:.1f} pts",
            color="#d62728")
    ax.set_xlabel("Spread error (final margin - spread, pts)")
    ax.set_ylabel("Number of games")
    ax.set_title("Figure 1b: Distribution of Point Spread Error")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1b_spread_error_hist.png", dpi=150)
    plt.close("all")


LABELS = {
    "games": ("Games analyzed", ""),
    "seasons": ("Seasons", ""),
    "slope": ("Slope m", ""),
    "intercept": ("Intercept b", "pts"),
    "R": ("Correlation R", ""),
    "R2": ("R^2", ""),
    "slope_p": ("p-value (slope)", ""),
    "mean_error": ("Mean error (bias)", "pts"),
    "MAE": ("Mean absolute error", "pts"),
    "median_abs_error": ("Median absolute error", "pts"),
    "RMSE": ("Root-mean-square error", "pts"),
    "favorite_win_pct": ("Favorite won outright", "%"),
    "within_3_pct": ("Within 3 pts of spread", "%"),
    "within_7_pct": ("Within 7 pts of spread", "%"),
}


def report(m):
    print("Question 1: How accurately does the point spread predict the final margin?")
    print("Method: least-squares linear fit  result = m * spread_line + b")
    print("-" * 72)
    for k, v in m.items():
        name, unit = LABELS[k]
        if isinstance(v, float):
            v = ("< 0.001" if v < 0.001 else f"{v:.3f}") if k == "slope_p" else f"{v:.3f}"
        print(f"  {name:<26}: {v} {unit}")
    print("-" * 72)


if __name__ == "__main__":
    d, fit, m = analyze(load_games())
    report(m)
    plot(d, fit)
    print("Figures saved to figures/fig1_spread_vs_margin.png and "
          "figures/fig1b_spread_error_hist.png")
