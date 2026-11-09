"""
load_data.py
Loads and cleans the nflverse games.csv file.

Every analysis script imports load_games() so all three questions
start from the same cleaned data.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "games.csv"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def load_games(regular_season_only=True):
    """Return a cleaned DataFrame of completed NFL games.

    Cleaning steps:
      1. Drop games that have not been played yet (no score).
      2. Optionally keep only regular-season games (game_type == "REG").
      3. Add helper columns used by more than one question.
    """
    df = pd.read_csv(DATA_PATH)

    # 1. Only games that have a final score
    df = df.dropna(subset=["home_score", "away_score"]).copy()

    # 2. Regular season only (playoffs have different incentives)
    if regular_season_only:
        df = df[df["game_type"] == "REG"].copy()

    # 3. Helper columns
    # result = home_score - away_score (positive means the home team won)
    # spread_line = how many points the home team was favored by
    df["spread_error"] = df["result"] - df["spread_line"]
    df["home_win"] = (df["result"] > 0).astype(int)
    df["is_outdoor"] = df["roof"].isin(["outdoors", "open"])

    return df


if __name__ == "__main__":
    games = load_games()
    print(f"Loaded {len(games)} completed regular-season games "
          f"({games['season'].min()}-{games['season'].max()})")
    print(games[["season", "home_team", "away_team", "result",
                 "spread_line", "total", "wind", "temp", "roof"]].head())
