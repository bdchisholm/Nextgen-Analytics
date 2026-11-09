"""
main.py - top-level script. Runs all three analyses and saves every figure
to the figures/ folder.

Usage (from the repo root):
    python src/main.py
"""

import q1_point_spread
import q2_home_field
import q3_weather
from load_data import load_games


def show(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))
    for k, v in metrics.items():
        print(f"  {k:>26}: {v:.4g}" if isinstance(v, float) else f"  {k:>26}: {v}")


def main():
    games = load_games()
    print(f"Loaded {len(games)} regular-season games "
          f"({games['season'].min()}-{games['season'].max()})")

    d1, fit, m1 = q1_point_spread.analyze(games)
    q1_point_spread.plot(d1, fit)
    show("Q1: Point spread accuracy", m1)

    by_season, trend, m2 = q2_home_field.analyze(games)
    q2_home_field.plot(by_season, trend)
    show("Q2: Home-field advantage", m2)

    d3, coef, m3 = q3_weather.analyze(games)
    q3_weather.plot(d3, coef)
    show("Q3: Weather and total points", m3)

    print("\nFigures saved to figures/")


if __name__ == "__main__":
    main()
