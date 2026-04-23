"""MoneyPuck data scraper for advanced hockey analytics."""

import io
import logging
import time
import zipfile
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

logger = logging.getLogger(__name__)

MONEYPUCK_BASE = "https://moneypuck.com/moneypuck/playerData"
CACHE_DIR = Path(__file__).parent / ".cache" / "moneypuck"

# Team abbreviation mapping (MoneyPuck -> standard)
TEAM_ABBREV_MAP = {
    "L.A": "LAK",
    "N.J": "NJD",
    "S.J": "SJS",
    "T.B": "TBL",
}


class MoneyPuckScraper:
    """Scraper for MoneyPuck advanced stats data."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        rate_limit: float = 1.0,
        use_cache: bool = True,
    ):
        """
        Initialize the scraper.

        Args:
            cache_dir: Directory to store cached data
            rate_limit: Minimum seconds between requests
            use_cache: Whether to use cached data
        """
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.use_cache = use_cache
        self.last_request_time = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NHL-Cup-Predictor/1.0"})

    def _rate_limit_wait(self) -> None:
        """Wait to respect rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

    def _download_csv(self, url: str, cache_name: str) -> pd.DataFrame:
        """Download a CSV file with caching."""
        cache_path = self.cache_dir / f"{cache_name}.parquet"

        if self.use_cache and cache_path.exists():
            logger.debug(f"Cache hit: {cache_name}")
            return pd.read_parquet(cache_path)

        self._rate_limit_wait()
        logger.info(f"Downloading: {url}")

        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            self.last_request_time = time.time()

            df = pd.read_csv(io.StringIO(response.text))

            if self.use_cache:
                df.to_parquet(cache_path)

            return df

        except requests.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            raise

    def _download_zip(self, url: str, cache_name: str) -> pd.DataFrame:
        """Download and extract a ZIP file containing CSV."""
        cache_path = self.cache_dir / f"{cache_name}.parquet"

        if self.use_cache and cache_path.exists():
            logger.debug(f"Cache hit: {cache_name}")
            return pd.read_parquet(cache_path)

        self._rate_limit_wait()
        logger.info(f"Downloading: {url}")

        try:
            response = self.session.get(url, timeout=120)
            response.raise_for_status()
            self.last_request_time = time.time()

            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
                if not csv_files:
                    raise ValueError(f"No CSV files found in {url}")

                # Read all CSVs and concatenate
                dfs = []
                for csv_file in csv_files:
                    with zf.open(csv_file) as f:
                        dfs.append(pd.read_csv(f))

                df = pd.concat(dfs, ignore_index=True)

            if self.use_cache:
                df.to_parquet(cache_path)

            return df

        except requests.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            raise

    def get_team_season_stats(
        self,
        season: int,
        situation: Literal["all", "5on5", "4on5", "5on4"] = "all",
        season_type: Literal["regular", "playoffs"] = "regular",
    ) -> pd.DataFrame:
        """
        Get team-level season summary stats.

        Args:
            season: Season end year (e.g., 2024 for 2023-24)
            situation: Game situation filter
            season_type: Regular season or playoffs

        Returns:
            DataFrame with team stats
        """
        url = f"{MONEYPUCK_BASE}/seasonSummary/{season}/{season_type}/teams.csv"
        cache_name = f"teams_{season}_{season_type}"

        df = self._download_csv(url, cache_name)

        # Filter by situation
        if situation != "all" and "situation" in df.columns:
            df = df[df["situation"] == situation].copy()

        # Standardize team abbreviations
        if "team" in df.columns:
            df["team"] = df["team"].replace(TEAM_ABBREV_MAP)

        return df

    def get_skater_stats(
        self,
        season: int,
        season_type: Literal["regular", "playoffs"] = "regular",
    ) -> pd.DataFrame:
        """
        Get skater-level stats for a season.

        Args:
            season: Season end year
            season_type: Regular season or playoffs

        Returns:
            DataFrame with skater stats
        """
        url = f"{MONEYPUCK_BASE}/seasonSummary/{season}/{season_type}/skaters.csv"
        cache_name = f"skaters_{season}_{season_type}"
        return self._download_csv(url, cache_name)

    def get_goalie_stats(
        self,
        season: int,
        season_type: Literal["regular", "playoffs"] = "regular",
    ) -> pd.DataFrame:
        """
        Get goalie-level stats for a season.

        Args:
            season: Season end year
            season_type: Regular season or playoffs

        Returns:
            DataFrame with goalie stats
        """
        url = f"{MONEYPUCK_BASE}/seasonSummary/{season}/{season_type}/goalies.csv"
        cache_name = f"goalies_{season}_{season_type}"

        df = self._download_csv(url, cache_name)

        # Standardize team abbreviations
        if "team" in df.columns:
            df["team"] = df["team"].replace(TEAM_ABBREV_MAP)

        return df

    def get_shots_data(self, season: int) -> pd.DataFrame:
        """
        Get shot-level data for a season.

        Warning: This is a large dataset (~100MB+ per season).

        Args:
            season: Season end year

        Returns:
            DataFrame with individual shot data
        """
        url = f"https://peter-tanner.com/moneypuck/downloads/shots_{season}.zip"
        cache_name = f"shots_{season}"
        return self._download_zip(url, cache_name)

    def get_game_level_team_stats(self, season: int) -> pd.DataFrame:
        """
        Compute game-level team stats from shots data.

        This aggregates shots data to get per-game advanced stats like
        CF%, FF%, xGF% for each team.

        Args:
            season: Season end year

        Returns:
            DataFrame with per-game team stats
        """
        logger.info(f"Computing game-level stats from shots for {season}...")

        shots = self.get_shots_data(season)

        # Group by game and team
        game_stats = (
            shots.groupby(["game_id", "teamCode", "isHomeTeam"])
            .agg(
                {
                    "shotWasOnGoal": "sum",
                    "goal": "sum",
                    "xGoal": "sum",
                    "event": "count",  # total shot attempts (corsi)
                }
            )
            .rename(
                columns={
                    "shotWasOnGoal": "shots_on_goal",
                    "goal": "goals",
                    "xGoal": "xGoals",
                    "event": "corsi",
                }
            )
            .reset_index()
        )

        # Calculate fenwick (unblocked shots)
        fenwick = shots[shots["shotWasOnGoal"] | (shots["event"] == "MISS")]
        fenwick_counts = (
            fenwick.groupby(["game_id", "teamCode"])
            .size()
            .reset_index(name="fenwick")
        )
        game_stats = game_stats.merge(
            fenwick_counts, on=["game_id", "teamCode"], how="left"
        )

        # Standardize team abbreviations
        game_stats["teamCode"] = game_stats["teamCode"].replace(TEAM_ABBREV_MAP)

        return game_stats

    def scrape_season(self, season: int) -> dict[str, pd.DataFrame]:
        """
        Scrape all available data for a season.

        Args:
            season: Season end year

        Returns:
            Dictionary of DataFrames by data type
        """
        logger.info(f"Scraping MoneyPuck data for {season-1}-{season} season...")

        data = {}

        # Team season summaries
        try:
            data["team_stats"] = self.get_team_season_stats(season)
            logger.info(f"  Team stats: {len(data['team_stats'])} records")
        except Exception as e:
            logger.warning(f"  Failed to get team stats: {e}")

        # Goalie stats
        try:
            data["goalie_stats"] = self.get_goalie_stats(season)
            logger.info(f"  Goalie stats: {len(data['goalie_stats'])} records")
        except Exception as e:
            logger.warning(f"  Failed to get goalie stats: {e}")

        return data

    def clear_cache(self) -> None:
        """Clear all cached data."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("MoneyPuck cache cleared")
