# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NHL Cup Predictor: ML model to predict Stanley Cup playoff outcomes using 12 seasons (2014-15 to 2025-26) of NHL game data and MoneyPuck advanced analytics.

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Data scraping (runs ~3-4 hours for all 12 seasons; uses cache on re-runs)
python -m src.scrape --seasons 2015-2026
python -m src.scrape --seasons 2024        # Single season
python -m src.scrape --seasons 2024 --no-cache --clear-cache  # Fresh pull

# Tests
pytest tests/                              # Unit tests only
pytest tests/ --run-slow                   # Include integration tests (network)
pytest tests/ -v --cov=src                 # With coverage

# Feature engineering (Phase 2)
python -m src.features                           # Process data/raw → data/processed
python -m src.features -v                        # Verbose output
python -m src.features --raw-data data/raw --output data/processed

# Prediction (Phase 4+)
python -m src.predict --season 2026 --simulations 10000
python -m src.predict --season 2026 --simulations 10000 --seed 12345  # Reproducible
```

## Architecture

```
src/
├── scrape/          # Data collection layer (Phase 1)
│   ├── nhl_api.py   # NHL API client (api-web.nhle.com) - games, boxscores, standings
│   ├── moneypuck.py # MoneyPuck client - advanced stats (CF%, FF%, xGF%, goalie stats)
│   └── pipeline.py  # Orchestrates both scrapers, outputs to data/raw/*.parquet
├── features/        # Feature engineering (Phase 2)
│   ├── engineering.py # Builds rolling stats, H2H records, rest days, goalie form, playoff exp
│   └── __main__.py   # CLI: python -m src.features
├── models/          # Model training (Phase 3)
└── simulate/        # Playoff Monte Carlo (Phase 4)
```

## Data Flow

1. **Raw data** (`data/raw/`): Parquet files from scrapers
   - `nhl_games.parquet`: Game results, shots, goalie stats per game
   - `moneypuck_team_stats.parquet`: Season-level advanced metrics by team
   - `moneypuck_goalie_stats.parquet`: Goalie performance metrics

2. **Processed data** (`data/processed/`): Feature-engineered datasets
   - `games_with_features.parquet`: One row per game with features and target `home_win`

## Features (Phase 2)

For each game, we compute (using only data from BEFORE the game):

1. **Rolling team statistics** (10/25/41-game windows):
   - Goals for/against per game
   - Shots for/against per game

2. **Head-to-head records**:
   - Win % in last 10 games between these two teams

3. **Rest days**:
   - Days since last game for each team

4. **Goalie form**:
   - Save % over goalie's last 10 starts

5. **Playoff experience**:
   - Playoff games played in recent seasons

All features use `.shift(1)` to prevent data leakage (game N features only use games 0..N-1).

## Key Design Decisions

- **No data leakage**: Features for game G only use data available before G
- **Fixed seed (42)** for model training; logged + overridable seed for Monte Carlo sims
- **Rate limiting**: 0.5s between NHL API calls, 1s between MoneyPuck downloads
- **Local cache**: JSON/Parquet in `src/scrape/.cache/` - re-runs are fast

## Data Sources

- **NHL API** (`api-web.nhle.com/v1`): Schedule, boxscores, standings, playoff bracket
- **MoneyPuck** (`moneypuck.com/data.htm`): CF%, FF%, xGF%, PDO, goalie GSAx

## Train/Val/Test Split

- Train: Seasons N-10 to N-2
- Validation: Season N-1
- Test: Season N (current season)

## Phase 5: Productionization Scope

1. **CLI interface**: `python -m src.predict --season 2026 --simulations 10000 --seed N`

2. **Daily refresh** (morning cron or GitHub Actions):
   - Scrape new game results
   - Re-run predictions
   - Generate updated HTML report

3. **Betting odds comparison** (The Odds API):
   - Fetch Cup futures + game lines from multiple sportsbooks
   - Compare model probabilities vs. market implied probabilities
   - Highlight value edges (model significantly disagrees with market)

4. **Injury tracking** (DailyFaceoff scraper):
   - Scrape current injury list (IR, day-to-day, out)
   - Flag impact level (star player vs. depth)
   - Show warnings in report + optional confidence adjustment

5. **HTML report output**:
   - Cup probability rankings for all playoff teams
   - Bracket visualization with win probabilities
   - Betting odds comparison table
   - Injury alerts
