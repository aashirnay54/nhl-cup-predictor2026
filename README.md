# NHL Cup Predictor

Machine learning model to predict Stanley Cup playoff outcomes using historical NHL data.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data Sources

- **NHL API**: Game-by-game results, team stats, goalie stats from `api-web.nhle.com`
- **MoneyPuck**: Advanced analytics (CF%, FF%, xGF%, etc.) from `moneypuck.com/data.htm`

## Usage

### Data Collection
```bash
# Pull all data for seasons 2014-15 through 2025-26
python -m src.scrape --seasons 2015-2026

# Pull specific season
python -m src.scrape --seasons 2026
```

### Prediction
```bash
# Run Cup simulation with 10,000 iterations
python -m src.predict --season 2026 --simulations 10000

# Reproduce a specific run
python -m src.predict --season 2026 --simulations 10000 --seed 12345
```

## Project Structure

```
nhl-cup-predictor/
├── data/
│   ├── raw/          # Raw API responses and CSVs as parquet
│   └── processed/    # Feature-engineered datasets
├── src/
│   ├── scrape/       # Data collection from NHL API and MoneyPuck
│   ├── features/     # Feature engineering pipeline
│   ├── models/       # Model training and evaluation
│   └── simulate/     # Playoff bracket Monte Carlo simulation
├── notebooks/        # Exploratory analysis
└── tests/            # Unit tests
```

## Models

1. **Single-game win probability**: Logistic regression baseline + XGBoost with isotonic calibration
2. **Series win probability**: Monte Carlo best-of-7 simulation with per-game home-ice adjustment

## Training Methodology

- Train: Seasons N-10 to N-2
- Validation: Season N-1
- Test: Season N
- Fixed random seed (42) for reproducibility

## Deployment Features (Phase 5)

- **Daily refresh**: Automated morning run via cron/GitHub Actions
- **Betting odds**: Integration with The Odds API to compare model vs. market
- **Injury tracking**: DailyFaceoff scraper for real-time injury alerts
- **HTML reports**: Bracket visualization, probabilities, value bets, injury warnings
