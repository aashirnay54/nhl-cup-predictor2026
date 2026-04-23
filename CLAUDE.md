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

# Prediction (Phase 4+)
python -m src.predict --season 2026 --simulations 10000
python -m src.predict --season 2026 --simulations 10000 --seed 12345  # Reproducible
```

## Architecture

```
src/
├── scrape/          # Data collection layer
│   ├── nhl_api.py   # NHL API client (api-web.nhle.com) - games, boxscores, standings
│   ├── moneypuck.py # MoneyPuck client - advanced stats (CF%, FF%, xGF%, goalie stats)
│   └── pipeline.py  # Orchestrates both scrapers, outputs to data/raw/*.parquet
├── features/        # Feature engineering (Phase 2)
├── models/          # Model training (Phase 3)
└── simulate/        # Playoff Monte Carlo (Phase 4)
```

## Data Flow

1. **Raw data** (`data/raw/`): Parquet files from scrapers
   - `nhl_games.parquet`: Game results, shots, goalie stats per game
   - `moneypuck_team_stats.parquet`: Season-level advanced metrics by team
   - `moneypuck_goalie_stats.parquet`: Goalie performance metrics

2. **Processed data** (`data/processed/`): Feature-engineered datasets
   - `games.parquet`: One row per game with rolling features and target `home_win`

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
