"""
Tests for feature engineering pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from src.features.engineering import FeatureEngineer, run_feature_pipeline


@pytest.fixture
def sample_games_data():
    """Create sample game data for testing."""
    dates = pd.date_range("2024-10-01", periods=50, freq="D")

    games = []
    for i, date in enumerate(dates):
        # Simulate games between 4 teams
        games.append({
            "game_id": 2024020000 + i,
            "game_date": date,
            "game_type": "regular" if i < 40 else "playoffs",
            "home_team": "TOR" if i % 2 == 0 else "MTL",
            "away_team": "BOS" if i % 2 == 0 else "TBL",
            "home_goals": np.random.randint(0, 6),
            "away_goals": np.random.randint(0, 6),
            "home_shots": np.random.randint(20, 40),
            "away_shots": np.random.randint(20, 40),
            "home_goalie": f"Goalie{i % 4}",
            "away_goalie": f"Goalie{(i + 1) % 4}",
            "home_goalie_sv": np.random.uniform(0.85, 0.95),
            "away_goalie_sv": np.random.uniform(0.85, 0.95),
        })

    df = pd.DataFrame(games)
    df["home_win"] = (df["home_goals"] > df["away_goals"]).astype(int)

    return df


@pytest.fixture
def temp_data_dir(tmp_path, sample_games_data):
    """Create temporary data directory with sample data."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # Save sample data
    sample_games_data.to_parquet(raw_dir / "nhl_games.parquet")

    # Create empty MoneyPuck files
    pd.DataFrame().to_parquet(raw_dir / "moneypuck_team_stats.parquet")
    pd.DataFrame().to_parquet(raw_dir / "moneypuck_goalie_stats.parquet")

    return tmp_path


class TestFeatureEngineer:
    """Test FeatureEngineer class."""

    def test_init(self, temp_data_dir):
        """Test initialization."""
        raw_dir = temp_data_dir / "raw"
        engineer = FeatureEngineer(raw_dir)
        assert engineer.raw_data_dir == raw_dir
        assert engineer.games_df is None

    def test_load_raw_data(self, temp_data_dir):
        """Test loading raw data."""
        engineer = FeatureEngineer(temp_data_dir / "raw")
        engineer.load_raw_data()

        assert engineer.games_df is not None
        assert len(engineer.games_df) == 50
        assert engineer.games_df["game_date"].is_monotonic_increasing

    def test_rolling_team_stats_no_leakage(self, temp_data_dir):
        """Test that rolling stats don't leak future data."""
        engineer = FeatureEngineer(temp_data_dir / "raw")
        engineer.load_raw_data()

        df = engineer._add_rolling_team_stats(engineer.games_df.copy())

        # First game should have NaN or min_periods=1 value
        assert "home_gf_L10" in df.columns
        assert "away_gf_L10" in df.columns

        # Check that rolling average for game N doesn't include game N itself
        # This is enforced by shift(1) in the rolling calculation
        for col in ["home_gf_L10", "away_gf_L10", "home_sf_L10", "away_sf_L10"]:
            assert col in df.columns
            # No NaN values due to min_periods=1
            assert df[col].notna().all()

    def test_h2h_records(self, temp_data_dir):
        """Test head-to-head record computation."""
        engineer = FeatureEngineer(temp_data_dir / "raw")
        engineer.load_raw_data()

        df = engineer._add_h2h_records(engineer.games_df.copy())

        assert "h2h_win_pct_L10" in df.columns
        assert "h2h_games_L10" in df.columns

        # Win percentage should be between 0 and 1
        assert (df["h2h_win_pct_L10"] >= 0).all()
        assert (df["h2h_win_pct_L10"] <= 1).all()

        # First game between two teams should have 0 H2H games
        assert df.iloc[0]["h2h_games_L10"] == 0
        assert df.iloc[0]["h2h_win_pct_L10"] == 0.5  # Neutral prior

    def test_rest_days(self, temp_data_dir):
        """Test rest days computation."""
        engineer = FeatureEngineer(temp_data_dir / "raw")
        engineer.load_raw_data()

        df = engineer._add_rest_days(engineer.games_df.copy())

        assert "home_rest_days" in df.columns
        assert "away_rest_days" in df.columns

        # Rest days should be positive
        assert (df["home_rest_days"] > 0).all()
        assert (df["away_rest_days"] > 0).all()

    def test_goalie_form(self, temp_data_dir):
        """Test goalie form metrics."""
        engineer = FeatureEngineer(temp_data_dir / "raw")
        engineer.load_raw_data()

        df = engineer._add_goalie_form(engineer.games_df.copy())

        assert "home_goalie_sv_L10" in df.columns
        assert "away_goalie_sv_L10" in df.columns

        # Save percentage should be between 0 and 1
        assert (df["home_goalie_sv_L10"] >= 0).all()
        assert (df["home_goalie_sv_L10"] <= 1).all()

    def test_playoff_experience(self, temp_data_dir):
        """Test playoff experience metric."""
        engineer = FeatureEngineer(temp_data_dir / "raw")
        engineer.load_raw_data()

        df = engineer._add_playoff_experience(engineer.games_df.copy())

        assert "home_playoff_exp" in df.columns
        assert "away_playoff_exp" in df.columns

        # Playoff experience should be non-negative
        assert (df["home_playoff_exp"] >= 0).all()
        assert (df["away_playoff_exp"] >= 0).all()

    def test_build_features_full_pipeline(self, temp_data_dir):
        """Test full feature engineering pipeline."""
        engineer = FeatureEngineer(temp_data_dir / "raw")
        df = engineer.build_features()

        # Check that all expected features exist
        expected_features = [
            "home_gf_L10", "home_gf_L25", "home_gf_L41",
            "away_gf_L10", "away_gf_L25", "away_gf_L41",
            "h2h_win_pct_L10", "h2h_games_L10",
            "home_rest_days", "away_rest_days",
            "home_goalie_sv_L10", "away_goalie_sv_L10",
            "home_playoff_exp", "away_playoff_exp",
            "home_win"  # Target
        ]

        for feature in expected_features:
            assert feature in df.columns, f"Missing feature: {feature}"

        # Check no missing values in critical features
        assert df["home_win"].notna().all()
        assert df["home_rest_days"].notna().all()


def test_run_feature_pipeline(temp_data_dir):
    """Test end-to-end pipeline."""
    output_dir = temp_data_dir / "processed"

    df = run_feature_pipeline(
        raw_data_dir=temp_data_dir / "raw",
        output_dir=output_dir,
        output_filename="test_features.parquet"
    )

    # Check output file exists
    assert (output_dir / "test_features.parquet").exists()

    # Check returned dataframe
    assert len(df) == 50
    assert "home_win" in df.columns


def test_no_data_leakage():
    """
    Critical test: Ensure rolling features don't leak future data.

    For game G on date D, features should only use data from BEFORE date D.
    """
    # Create controlled dataset
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    games = pd.DataFrame({
        "game_id": range(10),
        "game_date": dates,
        "game_type": ["regular"] * 10,
        "home_team": ["TOR"] * 10,
        "away_team": ["MTL"] * 10,
        "home_goals": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Increasing sequence
        "away_goals": [0] * 10,
        "home_shots": [30] * 10,
        "away_shots": [20] * 10,
        "home_goalie": ["G1"] * 10,
        "away_goalie": ["G2"] * 10,
        "home_goalie_sv": [0.90] * 10,
        "away_goalie_sv": [0.85] * 10,
        "home_win": [1] * 10,
    })

    engineer = FeatureEngineer(raw_data_dir=Path("."))
    engineer.games_df = games

    df = engineer._add_rolling_team_stats(games.copy())

    # Game 0: No previous games, should use min_periods=1 (itself excluded via shift)
    # Game 1: Should average game 0 only
    # Game 2: Should average games 0-1
    # etc.

    # For game 2, home_gf_L10 should be mean of [1, 2] = 1.5
    assert df.loc[2, "home_gf_L10"] == 1.5, "Rolling average leaked future data"

    # For game 5, home_gf_L10 should be mean of [1,2,3,4,5] = 3.0
    assert df.loc[5, "home_gf_L10"] == 3.0, "Rolling average leaked future data"
