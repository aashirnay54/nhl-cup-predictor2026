"""NHL API scraper for game data and team stats."""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api-web.nhle.com/v1"
CACHE_DIR = Path(__file__).parent / ".cache"


class NHLAPIScraper:
    """Scraper for NHL API data with rate limiting and caching."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        rate_limit: float = 0.5,
        use_cache: bool = True,
    ):
        """
        Initialize the scraper.

        Args:
            cache_dir: Directory to store cached responses
            rate_limit: Minimum seconds between requests
            use_cache: Whether to use cached responses
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

    def _cache_path(self, endpoint: str) -> Path:
        """Get cache file path for an endpoint."""
        safe_name = endpoint.replace("/", "_").strip("_") + ".json"
        return self.cache_dir / safe_name

    def _get(self, endpoint: str) -> dict[str, Any]:
        """
        Make a GET request with caching and rate limiting.

        Args:
            endpoint: API endpoint (without base URL)

        Returns:
            JSON response as dictionary
        """
        cache_path = self._cache_path(endpoint)

        if self.use_cache and cache_path.exists():
            logger.debug(f"Cache hit: {endpoint}")
            return json.loads(cache_path.read_text())

        self._rate_limit_wait()
        url = f"{BASE_URL}{endpoint}"
        logger.debug(f"Fetching: {url}")

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            self.last_request_time = time.time()

            data = response.json()

            if self.use_cache:
                cache_path.write_text(json.dumps(data))

            return data

        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise

    def get_schedule(self, date: str) -> dict[str, Any]:
        """
        Get schedule for a specific date.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            Schedule data including games
        """
        return self._get(f"/schedule/{date}")

    def get_season_schedule(self, season: int) -> list[dict[str, Any]]:
        """
        Get all games for a season.

        Args:
            season: Season end year (e.g., 2024 for 2023-24 season)

        Returns:
            List of all games in the season
        """
        # NHL seasons run from October to June
        # Season 2024 = 2023-10-01 to 2024-06-30
        start_date = datetime(season - 1, 10, 1)
        end_date = datetime(season, 6, 30)

        all_games = []
        current_date = start_date

        logger.info(f"Fetching schedule for {season-1}-{season} season...")

        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            try:
                schedule = self.get_schedule(date_str)
                game_week = schedule.get("gameWeek", [])

                for day in game_week:
                    for game in day.get("games", []):
                        # Only include completed games
                        if game.get("gameState") in ("OFF", "FINAL"):
                            all_games.append(game)

                # Jump to next week if we got a full week
                if game_week:
                    last_date = game_week[-1].get("date", date_str)
                    current_date = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
                else:
                    current_date += timedelta(days=7)

            except requests.RequestException:
                current_date += timedelta(days=1)
                continue

        # Deduplicate by game ID
        seen_ids = set()
        unique_games = []
        for game in all_games:
            game_id = game.get("id")
            if game_id and game_id not in seen_ids:
                seen_ids.add(game_id)
                unique_games.append(game)

        logger.info(f"Found {len(unique_games)} games for {season-1}-{season} season")
        return unique_games

    def get_boxscore(self, game_id: int) -> dict[str, Any]:
        """
        Get boxscore for a specific game.

        Args:
            game_id: NHL game ID

        Returns:
            Boxscore data
        """
        return self._get(f"/gamecenter/{game_id}/boxscore")

    def get_standings(self, date: str) -> dict[str, Any]:
        """
        Get standings for a specific date.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            Standings data
        """
        return self._get(f"/standings/{date}")

    def get_playoff_bracket(self, season: int) -> dict[str, Any]:
        """
        Get playoff bracket for a season.

        Args:
            season: Season end year (e.g., 2024 for 2023-24 playoffs)

        Returns:
            Playoff bracket data
        """
        return self._get(f"/playoff-bracket/{season}")

    def scrape_season_games(self, season: int) -> pd.DataFrame:
        """
        Scrape all games for a season with boxscore stats.

        Args:
            season: Season end year

        Returns:
            DataFrame with game-level data
        """
        from tqdm import tqdm

        games = self.get_season_schedule(season)
        records = []

        logger.info(f"Fetching boxscores for {len(games)} games...")

        for game in tqdm(games, desc=f"Season {season}"):
            game_id = game.get("id")
            if not game_id:
                continue

            try:
                boxscore = self.get_boxscore(game_id)
                record = self._parse_boxscore(game, boxscore)
                if record:
                    records.append(record)
            except requests.RequestException as e:
                logger.warning(f"Failed to get boxscore for game {game_id}: {e}")
                continue

        df = pd.DataFrame(records)
        logger.info(f"Scraped {len(df)} games for {season-1}-{season} season")
        return df

    def _parse_boxscore(
        self, game: dict[str, Any], boxscore: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Parse game and boxscore data into a flat record."""
        try:
            game_id = game.get("id")
            game_date = game.get("startTimeUTC", "")[:10]
            game_type = game.get("gameType", 0)  # 2=regular, 3=playoff

            home_team = game.get("homeTeam", {})
            away_team = game.get("awayTeam", {})

            # Get team stats from boxscore
            home_stats = boxscore.get("homeTeam", {})
            away_stats = boxscore.get("awayTeam", {})

            # Get boxscore details
            box_home = boxscore.get("boxscore", {}).get("teamGameStats", [])
            box_away = boxscore.get("boxscore", {}).get("teamGameStats", [])

            # Parse team game stats if available
            home_sog = home_stats.get("sog", 0)
            away_sog = away_stats.get("sog", 0)

            # Find goalie stats
            home_goalies = boxscore.get("playerByGameStats", {}).get("homeTeam", {}).get("goalies", [])
            away_goalies = boxscore.get("playerByGameStats", {}).get("awayTeam", {}).get("goalies", [])

            home_starter = home_goalies[0] if home_goalies else {}
            away_starter = away_goalies[0] if away_goalies else {}

            record = {
                "game_id": game_id,
                "date": game_date,
                "season": game.get("season", 0),
                "game_type": game_type,
                "is_playoff": game_type == 3,
                # Home team
                "home_team_id": home_team.get("id"),
                "home_team": home_team.get("abbrev", ""),
                "home_goals": home_team.get("score", 0),
                "home_sog": home_sog,
                # Away team
                "away_team_id": away_team.get("id"),
                "away_team": away_team.get("abbrev", ""),
                "away_goals": away_team.get("score", 0),
                "away_sog": away_sog,
                # Goalie info
                "home_goalie_id": home_starter.get("playerId"),
                "home_goalie_saves": home_starter.get("saves", 0),
                "home_goalie_shots_against": home_starter.get("shotsAgainst", 0),
                "away_goalie_id": away_starter.get("playerId"),
                "away_goalie_saves": away_starter.get("saves", 0),
                "away_goalie_shots_against": away_starter.get("shotsAgainst", 0),
                # Result
                "home_win": int(home_team.get("score", 0) > away_team.get("score", 0)),
            }

            return record

        except Exception as e:
            logger.warning(f"Failed to parse game {game.get('id')}: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear all cached responses."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache cleared")
