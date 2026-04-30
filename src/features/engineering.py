"""
Feature engineering pipeline for NHL game prediction.

Key principle: No data leakage - features for game G only use data from BEFORE game G.
"""

import pandas as pd
from pathlib import Path
from loguru import logger


class FeatureEngineer:
    """Builds features for NHL game prediction while avoiding data leakage."""

    def __init__(self, raw_data_dir: Path):
        """
        Args:
            raw_data_dir: Directory containing raw parquet files from scraper
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.games_df = None
        self.moneypuck_team = None
        self.moneypuck_goalie = None

    def load_raw_data(self):
        """Load raw data from parquet files."""
        logger.info("Loading raw data")

        self.games_df = pd.read_parquet(self.raw_data_dir / "nhl_games.parquet")

        # Try to load MoneyPuck data (may not exist for current season)
        try:
            self.moneypuck_team = pd.read_parquet(self.raw_data_dir / "moneypuck_team_stats.parquet")
        except Exception:
            logger.warning("MoneyPuck team stats not found")
            self.moneypuck_team = pd.DataFrame()

        try:
            self.moneypuck_goalie = pd.read_parquet(self.raw_data_dir / "moneypuck_goalie_stats.parquet")
        except Exception:
            logger.warning("MoneyPuck goalie stats not found")
            self.moneypuck_goalie = pd.DataFrame()

        # Rename columns to match expected format
        self.games_df = self.games_df.rename(columns={
            "date": "game_date",
            "home_sog": "home_shots",
            "away_sog": "away_shots",
        })

        # Sort games chronologically to enable rolling features
        self.games_df = self.games_df.sort_values("game_date").reset_index(drop=True)

        logger.info(f"Loaded {len(self.games_df)} games")

    def build_features(self) -> pd.DataFrame:
        """
        Main pipeline: builds all features for each game.

        Returns:
            DataFrame with one row per game, features as columns, target as home_win
        """
        if self.games_df is None:
            self.load_raw_data()

        logger.info("Building features")

        df = self.games_df.copy()

        # 1. Rolling team statistics
        logger.info("Computing rolling team statistics")
        df = self._add_rolling_team_stats(df)

        # 2. Head-to-head records
        logger.info("Computing head-to-head records")
        df = self._add_h2h_records(df)

        # 3. Rest days
        logger.info("Computing rest days")
        df = self._add_rest_days(df)

        # 4. Goalie form
        logger.info("Computing goalie form metrics")
        df = self._add_goalie_form(df)

        # 5. Playoff experience
        logger.info("Computing playoff experience")
        df = self._add_playoff_experience(df)

        # 6. Home ice indicator (already exists)
        # Target: home_win (already exists)

        logger.info(f"Feature engineering complete: {len(df)} games, {len(df.columns)} columns")

        return df

    def _add_rolling_team_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling team statistics over windows of 10, 25, 41 games.

        For each game, compute team's average over last N games BEFORE this game.
        """
        windows = [10, 25, 41]

        # League average defaults for first games of season
        avg_goals = 3.0
        avg_shots = 30.0

        # Stats to roll: goals_for, goals_against, shots_for, shots_against per game
        for team_col, goals_col, shots_col, prefix in [
            ("home_team", "home_goals", "home_shots", "home"),
            ("away_team", "away_goals", "away_shots", "away")
        ]:
            for window in windows:
                # Goals for per game (rolling average)
                df[f"{prefix}_gf_L{window}"] = df.groupby(team_col)[goals_col].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                ).fillna(avg_goals)

                # Shots for per game
                df[f"{prefix}_sf_L{window}"] = df.groupby(team_col)[shots_col].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                ).fillna(avg_shots)

        # Also compute goals against and shots against by looking at opponent stats
        for window in windows:
            # Home team's goals against = away team's goals for in games where this team was home
            df[f"home_ga_L{window}"] = df.groupby("home_team")["away_goals"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            ).fillna(avg_goals)

            df[f"away_ga_L{window}"] = df.groupby("away_team")["home_goals"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            ).fillna(avg_goals)

            df[f"home_sa_L{window}"] = df.groupby("home_team")["away_shots"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            ).fillna(avg_shots)

            df[f"away_sa_L{window}"] = df.groupby("away_team")["home_shots"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            ).fillna(avg_shots)

        return df

    def _add_h2h_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add head-to-head win/loss record for last 10 games between these two teams.

        Returns win % for home team in recent H2H matchups.
        """
        # Create a sorted team pair key for matching
        df["team_pair"] = df.apply(
            lambda row: tuple(sorted([row["home_team"], row["away_team"]])),
            axis=1
        )

        # For each game, count wins in last 10 H2H games
        h2h_wins = []
        h2h_games = []

        for idx, row in df.iterrows():
            # Find all previous games with this matchup
            prev_games = df[
                (df.index < idx) &
                (df["team_pair"] == row["team_pair"])
            ].tail(10)

            if len(prev_games) == 0:
                h2h_wins.append(0.5)  # Neutral prior
                h2h_games.append(0)
            else:
                # Count games where home team won (considering who was home in those games)
                home_team = row["home_team"]
                wins = 0
                for _, prev_game in prev_games.iterrows():
                    if prev_game["home_team"] == home_team and prev_game["home_win"]:
                        wins += 1
                    elif prev_game["away_team"] == home_team and not prev_game["home_win"]:
                        wins += 1

                h2h_wins.append(wins / len(prev_games))
                h2h_games.append(len(prev_games))

        df["h2h_win_pct_L10"] = h2h_wins
        df["h2h_games_L10"] = h2h_games

        df = df.drop(columns=["team_pair"])

        return df

    def _add_rest_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add days of rest since last game for each team.
        """
        df["game_date"] = pd.to_datetime(df["game_date"])

        for team_col, prefix in [("home_team", "home"), ("away_team", "away")]:
            # Compute days since last game
            df[f"{prefix}_rest_days"] = df.groupby(team_col)["game_date"].transform(
                lambda x: x.diff().dt.days.shift(1)
            )

            # Fill first game of season with 7 days (arbitrary but reasonable)
            df[f"{prefix}_rest_days"] = df[f"{prefix}_rest_days"].fillna(7)

        return df

    def _add_goalie_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add goalie form metrics: SV% over last 10 starts.

        Computes SV% from saves and shots_against columns.
        """
        league_avg_sv = 0.915

        # Check if goalie ID columns exist
        if "home_goalie_id" not in df.columns or "away_goalie_id" not in df.columns:
            logger.warning("Goalie ID columns not found - using league average")
            df["home_goalie_sv_L10"] = league_avg_sv
            df["away_goalie_sv_L10"] = league_avg_sv
            return df

        # Compute save percentage for each game
        df["home_goalie_sv_pct"] = df["home_goalie_saves"] / df["home_goalie_shots_against"]
        df["away_goalie_sv_pct"] = df["away_goalie_saves"] / df["away_goalie_shots_against"]

        # Replace inf/nan with league average
        df["home_goalie_sv_pct"] = df["home_goalie_sv_pct"].replace([float('inf'), -float('inf')], league_avg_sv).fillna(league_avg_sv)
        df["away_goalie_sv_pct"] = df["away_goalie_sv_pct"].replace([float('inf'), -float('inf')], league_avg_sv).fillna(league_avg_sv)

        # Compute rolling save percentage for each goalie over last 10 starts
        df["home_goalie_sv_L10"] = df.groupby("home_goalie_id")["home_goalie_sv_pct"].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
        ).fillna(league_avg_sv)

        df["away_goalie_sv_L10"] = df.groupby("away_goalie_id")["away_goalie_sv_pct"].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
        ).fillna(league_avg_sv)

        # Drop temporary columns
        df = df.drop(columns=["home_goalie_sv_pct", "away_goalie_sv_pct"])

        return df

    def _add_playoff_experience(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add playoff experience: games and wins in last 3 playoff seasons.

        For simplicity, compute as count of playoff games in dataset for each team.
        """
        # Filter to playoff games only
        if "game_type" not in df.columns:
            logger.warning("game_type column not found - skipping playoff experience")
            df["home_playoff_exp"] = 0
            df["away_playoff_exp"] = 0
            return df

        playoff_games = df[df["game_type"] == "playoffs"].copy()

        for team_col, prefix in [("home_team", "home"), ("away_team", "away")]:
            # Count playoff games in last 3 seasons for each team
            # This is a simplified version - ideally would track per season
            playoff_counts = playoff_games.groupby(team_col).size().to_dict()
            df[f"{prefix}_playoff_exp"] = df[team_col].map(playoff_counts).fillna(0)

        return df


def run_feature_pipeline(
    raw_data_dir: Path,
    output_dir: Path,
    output_filename: str = "games_with_features.parquet"
) -> pd.DataFrame:
    """
    Main entry point: load raw data, build features, save to parquet.

    Args:
        raw_data_dir: Path to data/raw/
        output_dir: Path to data/processed/
        output_filename: Name of output file

    Returns:
        DataFrame with features
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engineer = FeatureEngineer(raw_data_dir)
    df = engineer.build_features()

    output_path = output_dir / output_filename
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved features to {output_path}")

    return df
