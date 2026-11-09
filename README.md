# Predicting NFL Outcomes Using Historical Statistics and Python

**ENGR 315 – Computational Methods in Engineering · James Madison University · Fall 2026**

## Team Name and Contributors

**Team:** NEXTGEN ANALYTICS

| Name | GitHub |
|------|--------|
| Dylan Bedois | `@<dylan-github-username>` |
| Brian Chisholm | `@bdchisholm` |

---

## Objective and Motivation

Our team has been interested in analytics in professional sports for years, and we want to work toward software that can predict NFL game outcomes. Every season the NFL produces a large amount of data: points scored, betting lines, stadium conditions, rest days and more. We want to find out whether that historical data shows patterns that can be used to make predictions about future games, and what contributes to teams winning.

### Dataset

| Item | Detail |
|------|--------|
| Name | NFL schedules and results (`games.csv`) |
| Source | nflverse – <https://github.com/nflverse/nfldata/blob/master/data/games.csv> |
| Maintainer | nflverse, an open-source community that collects NFL data for public analysis |
| Size | ~2.2 MB CSV, one row per game, 46 columns |
| Coverage | Every NFL game from the 1999 season to the present (7,548 rows; 7,102 completed regular-season games) |
| Origin of values | Scores, dates and stadium info come from official NFL game records; betting columns come from published sportsbook closing lines |
| Copy in this repo | [`data/games.csv`](data/games.csv) (snapshot downloaded November 2026) |

**Columns we use**

| Column | Meaning |
|--------|---------|
| `season`, `game_type`, `week` | When the game was played; `REG` = regular season |
| `home_score`, `away_score` | Final score for each team |
| `result` | Home score − away score (positive = home team won) |
| `total` | Combined points scored |
| `spread_line` | Points the home team was favored by before kickoff |
| `total_line` | Betting over/under line for combined points |
| `location` | `Home` or `Neutral` site |
| `roof` | `outdoors`, `open`, `closed` or `dome` |
| `temp`, `wind` | Temperature (°F) and wind speed (mph) at kickoff (outdoor games only) |
| `home_rest`, `away_rest` | Days of rest for each team |

> Because this repository is not one of the instructor's suggested data sites, we will confirm with the instructor that it is acceptable.

### Questions, Expected Form of the Answers, and Benefits

| # | Question | Form of the expected answer | Potential benefit |
|---|----------|-----------------------------|-------------------|
| **Q1** | How accurately does the pre-game point spread predict the final margin of victory? | A fitted line (slope, intercept), R and R², and the typical error of the spread in points (MAE / RMSE) | Gives a baseline that any prediction model we build later must beat |
| **Q2** | How large is home-field advantage, and has it changed over time? | Average home margin in points with a 95% confidence interval, p-values for "is it > 0?" and "is it smaller in recent seasons?", and a trend in points per season | Tests the common "home field is worth 3 points" claim; useful input for predictions |
| **Q3** | How do wind speed and temperature at kickoff affect total points in outdoor games? | Regression coefficients (points lost per mph of wind, per °F), p-values, and the difference in average points between windy and calm games | Shows whether weather should be a feature in a scoring prediction |

---

## Introduction / Context

A few terms are needed to follow the methods:

- **Point spread (`spread_line`)** – the sportsbooks' pre-game prediction of the margin. A spread of +3 means the home team is expected to win by 3. If the spread were a perfect predictor, `result = spread_line` for every game.
- **Final margin (`result`)** – what actually happened, home score minus away score.
- **Home-field advantage** – the average amount by which home teams outscore visitors. Neutral-site games (e.g. international games) are removed because neither team is really at home.
- **Outdoor games** – only games with `roof` = `outdoors` or `open` have weather recorded, so Q3 uses only those.
- **2020 season** – most games were played with few or no fans because of COVID-19, which makes it a natural experiment for Q2.

---

## Implementation Details

### Overall workflow

```
games.csv ──► load_data.py ──► cleaned DataFrame ─┬─► q1_point_spread.py ──► metrics + Fig 1, 1b
              (drop unplayed games,               ├─► q2_home_field.py   ──► metrics + Fig 2, 2b
               keep regular season,               └─► q3_weather.py      ──► metrics + Fig 3, 3b
               add helper columns)
                                   main.py runs all three and saves every figure to figures/
```

**Common cleaning (all questions)** – done in [`src/load_data.py`](src/load_data.py):
1. Drop games without a final score (future games in the current season).
2. Keep only regular-season games (`game_type == "REG"`); playoff games have different incentives and much smaller sample sizes.
3. Add helper columns: `spread_error = result − spread_line`, `home_win` (1/0), `is_outdoor` (roof is `outdoors` or `open`).

### Python packages

| Package | Used for |
|---------|----------|
| `pandas` | Reading the CSV, filtering, grouping by season / wind bin |
| `numpy` | Array math, least-squares solution of the regression (`np.linalg.lstsq`), error metrics |
| `scipy.stats` | `linregress`, `ttest_1samp`, `ttest_ind` (Welch), `pearsonr`, t-distribution confidence intervals |
| `matplotlib` | All plots |

Install with `pip install -r requirements.txt`. No packages beyond the course's standard scientific Python stack are needed.

---

### Q1 – Point spread accuracy (data fitting)

**Data:** all completed regular-season games with a spread (`spread_line`, `result`).

**Method**
1. Least-squares linear fit: `result = m · spread_line + b` (`scipy.stats.linregress`).
2. Compute the spread's prediction error for every game: `error = result − spread_line`.
3. Summarize the error and how often the favorite won.

**Metrics**

| Metric | What it tells us |
|--------|------------------|
| Slope *m* and intercept *b* | *m* ≈ 1 and *b* ≈ 0 means the spread is unbiased on average |
| Correlation R and R² | How much of the variation in final margins the spread explains |
| MAE and RMSE (points) | How far off the spread typically is in a single game |
| Favorite win % | How often the team favored by the spread won outright |
| % of games within 3 / 7 points of the spread | How often the spread is "close" |

**Plots**
- *Figure 1* – scatter of spread vs. final margin with the fitted line and the ideal *y = x* line.
- *Figure 1b* – histogram of spread error, to check whether errors are centered at 0 and roughly normal.

**Code:** [`src/q1_point_spread.py`](src/q1_point_spread.py)

---

### Q2 – Home-field advantage over time (statistical analysis)

**Data:** completed regular-season games with `location == "Home"` (neutral sites removed).

**Method**
1. **One-sample t-test** on `result`: H₀: mean home margin = 0 vs. H₁: mean home margin > 0.
2. **95% confidence interval** for the mean home margin (t-distribution).
3. **Two-sample Welch t-test** comparing 1999–2018 with 2019–present (H₀: equal means).
4. Group by `season` and fit a **linear trend** of season-average home margin vs. season.

**Metrics**

| Metric | What it tells us |
|--------|------------------|
| Mean home margin and 95% CI (points) | Size of home-field advantage |
| Home win % | Same idea, expressed as a win rate |
| t statistic and p-value (one-sample) | Whether home-field advantage is real |
| Era means, Welch t and p-value | Whether it has shrunk recently |
| Trend slope (points per season) and p-value | Long-term direction |

**Plots**
- *Figure 2* – bar chart of average home margin by season, with the trend line and the 2020 season highlighted.
- *Figure 2b* – home win percentage by season.

**Code:** [`src/q2_home_field.py`](src/q2_home_field.py)

---

### Q3 – Weather and total points (statistical analysis + data fitting)

**Data:** completed regular-season games with `roof` in {`outdoors`, `open`} and non-missing `wind` and `temp`.

**Method**
1. **Multiple linear regression:** `total = b₀ + b₁·wind + b₂·temp`, solved with `numpy.linalg.lstsq`; standard errors and p-values computed from the residuals.
2. **Pearson correlation** of `total` with `wind` and with `temp` separately.
3. **Two-sample Welch t-test**: windy games (wind ≥ 15 mph) vs. calm games (< 15 mph).
4. **(Phase 2 extension)** Compare `total` with the betting over/under (`total_line`) in windy games to see whether the betting market already accounts for weather.

**Metrics**

| Metric | What it tells us |
|--------|------------------|
| b₁ (points per mph) and b₂ (points per °F) with p-values | Size and significance of each weather effect |
| R² | How much of the scoring variation weather explains |
| Pearson r for wind and temp | Strength and direction of each relationship |
| Mean total for windy vs. calm, Welch t and p | Simple, easy-to-explain version of the wind effect |

**Plots**
- *Figure 3* – average total points by wind-speed bin (0–4, 5–9, 10–14, 15–19, 20+ mph) with 95% confidence-interval error bars and sample sizes.
- *Figure 3b* – scatter of temperature vs. total points with the regression line (wind held at its mean).

**Code:** [`src/q3_weather.py`](src/q3_weather.py)

---

### How to run

```bash
git clone https://github.com/<owner>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
python src/main.py          # prints all metrics and saves figures to figures/
```

Each question can also be run on its own, e.g. `python src/q2_home_field.py`.

### Preliminary figures (first run of the code)

| Q1 | Q2 | Q3 |
|----|----|----|
| ![Figure 1](figures/fig1_spread_vs_margin.png) | ![Figure 2](figures/fig2_home_field_by_season.png) | ![Figure 3](figures/fig3_points_by_wind.png) |

### Code snippets / block diagrams

*Placeholder – screenshots of key code and block diagrams will be added in the coming phases.*

---

## Results (Answers to the Questions)

*To be completed in a later phase.*

---

## Conclusions

*To be completed in a later phase.*

---

## Actions That Need to Be Completed

*To be completed in a later phase. Current checklist:*

- [x] Select dataset and questions (proposal)
- [x] Identify methods, metrics, packages and plots for each question
- [x] Create GitHub repo with README and add all team members as collaborators
- [x] Write data loading / cleaning script
- [x] First working version of Q1, Q2 and Q3 analysis scripts
- [ ] Confirm with instructor that the nflverse dataset is acceptable
- [ ] Review preliminary numbers and finalize statistical tests
- [ ] Q3 extension: compare windy-game totals with the betting over/under
- [ ] Write Results section with final figures and numbers
- [ ] Write Conclusions section
- [ ] Add code screenshots / block diagram
- [ ] Prepare final presentation

**Issues the group is facing**

- The dataset is not on the instructor's suggested list – approval pending.
- The current season (2026) is still in progress, so its numbers change each week; we may freeze the analysis at the end of the 2025 season.
- Our original motivation mentioned yards and turnovers, but this dataset does not include them (would need much larger play-by-play data).

---

## References / Source Files

1. nflverse, *nfldata – games.csv* (NFL schedules and results). <https://github.com/nflverse/nfldata>
2. nflverse project. <https://github.com/nflverse>
3. SciPy documentation – `scipy.stats`. <https://docs.scipy.org/doc/scipy/reference/stats.html>
4. pandas documentation. <https://pandas.pydata.org/docs/>
5. Matplotlib documentation. <https://matplotlib.org/stable/>
6. NumPy documentation – `numpy.linalg.lstsq`. <https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html>
