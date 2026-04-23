"""Data scraping pipeline coordinating NHL API and MoneyPuck data."""

import logging
from pathlib import Path

import pandas as pd

from .moneypuck import MoneyPuckScraper
from .nhl_api import NHLAPIScraper

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def run_scrape_pipeline(
    seasons: list[int],
    output_dir: Path | None = None,
    use_cache: bool = True,
    include_shots: bool = False,
) -> dict[str, Path]:
    """
    Run the full data scraping pipeline.

    Args:
        seasons: List of season end years to scrape (e.g., [2024, 2025])
        output_dir: Directory to save output files (default: data/raw)
        use_cache: Whether to use cached API responses
        include_shots: Whether to download shot-level data (large files)

    Returns:
        Dictionary mapping data type to output file path
    """
    output_dir = output_dir or DATA_DIR / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    nhl_scraper = NHLAPIScraper(use_cache=use_cache)
    mp_scraper = MoneyPuckScraper(use_cache=use_cache)

    output_files = {}

    # Scrape NHL API game data
    logger.info("=" * 60)
    logger.info("SCRAPING NHL API GAME DATA")
    logger.info("=" * 60)

    all_games = []
    for season in seasons:
        try:
            df = nhl_scraper.scrape_season_games(season)
            df["season_year"] = season
            all_games.append(df)
        except Exception as e:
            logger.error(f"Failed to scrape NHL API for {season}: {e}")

    if all_games:
        games_df = pd.concat(all_games, ignore_index=True)
        games_path = output_dir / "nhl_games.parquet"
        games_df.to_parquet(games_path)
        output_files["games"] = games_path
        logger.info(f"Saved {len(games_df)} games to {games_path}")

    # Scrape MoneyPuck team stats
    logger.info("=" * 60)
    logger.info("SCRAPING MONEYPUCK TEAM STATS")
    logger.info("=" * 60)

    all_team_stats = []
    all_goalie_stats = []

    for season in seasons:
        try:
            data = mp_scraper.scrape_season(season)

            if "team_stats" in data:
                data["team_stats"]["season_year"] = season
                all_team_stats.append(data["team_stats"])

            if "goalie_stats" in data:
                data["goalie_stats"]["season_year"] = season
                all_goalie_stats.append(data["goalie_stats"])

        except Exception as e:
            logger.error(f"Failed to scrape MoneyPuck for {season}: {e}")

    if all_team_stats:
        team_stats_df = pd.concat(all_team_stats, ignore_index=True)
        team_stats_path = output_dir / "moneypuck_team_stats.parquet"
        team_stats_df.to_parquet(team_stats_path)
        output_files["team_stats"] = team_stats_path
        logger.info(f"Saved team stats to {team_stats_path}")

    if all_goalie_stats:
        goalie_stats_df = pd.concat(all_goalie_stats, ignore_index=True)
        goalie_stats_path = output_dir / "moneypuck_goalie_stats.parquet"
        goalie_stats_df.to_parquet(goalie_stats_path)
        output_files["goalie_stats"] = goalie_stats_path
        logger.info(f"Saved goalie stats to {goalie_stats_path}")

    # Optionally scrape shot-level data
    if include_shots:
        logger.info("=" * 60)
        logger.info("SCRAPING MONEYPUCK SHOT DATA (this may take a while)")
        logger.info("=" * 60)

        all_shots = []
        for season in seasons:
            try:
                shots_df = mp_scraper.get_shots_data(season)
                shots_df["season_year"] = season
                all_shots.append(shots_df)
            except Exception as e:
                logger.error(f"Failed to scrape shots for {season}: {e}")

        if all_shots:
            shots_df = pd.concat(all_shots, ignore_index=True)
            shots_path = output_dir / "moneypuck_shots.parquet"
            shots_df.to_parquet(shots_path)
            output_files["shots"] = shots_path
            logger.info(f"Saved shot data to {shots_path}")

    logger.info("=" * 60)
    logger.info("SCRAPING COMPLETE")
    logger.info("=" * 60)
    for name, path in output_files.items():
        logger.info(f"  {name}: {path}")

    return output_files


def parse_season_range(season_arg: str) -> list[int]:
    """
    Parse a season range string into list of season years.

    Args:
        season_arg: Season range like "2015-2025" or single season "2024"

    Returns:
        List of season end years

    Examples:
        "2015-2025" -> [2015, 2016, ..., 2025]
        "2024" -> [2024]
        "2024-2025" -> [2024, 2025]
    """
    if "-" in season_arg:
        parts = season_arg.split("-")
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            # Handle both "2015-2025" and "2024-2025" formats
            if end < start:
                # Probably "2024-25" format, convert to "2025"
                end = int(parts[0][:2] + parts[1])
            return list(range(start, end + 1))
    return [int(season_arg)]
