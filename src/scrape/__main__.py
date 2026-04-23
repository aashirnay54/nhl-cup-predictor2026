"""CLI entry point for data scraping.

Usage:
    python -m src.scrape --seasons 2015-2025
    python -m src.scrape --seasons 2024-2025 --no-cache
    python -m src.scrape --seasons 2024 --include-shots
"""

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import parse_season_range, run_scrape_pipeline


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scrape NHL game data and MoneyPuck advanced stats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.scrape --seasons 2015-2025     # Scrape 11 seasons
  python -m src.scrape --seasons 2024          # Scrape single season
  python -m src.scrape --seasons 2024 --no-cache --include-shots
        """,
    )

    parser.add_argument(
        "--seasons",
        type=str,
        required=True,
        help="Season(s) to scrape. Format: '2024' or '2015-2025'",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/raw)",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (re-fetch all data)",
    )

    parser.add_argument(
        "--include-shots",
        action="store_true",
        help="Include shot-level data (large downloads)",
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached data before scraping",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Parse seasons
    try:
        seasons = parse_season_range(args.seasons)
        logger.info(f"Seasons to scrape: {seasons}")
    except ValueError as e:
        logger.error(f"Invalid season format: {e}")
        return 1

    # Clear cache if requested
    if args.clear_cache:
        from .moneypuck import MoneyPuckScraper
        from .nhl_api import NHLAPIScraper

        logger.info("Clearing caches...")
        NHLAPIScraper().clear_cache()
        MoneyPuckScraper().clear_cache()

    # Run pipeline
    try:
        output_files = run_scrape_pipeline(
            seasons=seasons,
            output_dir=args.output_dir,
            use_cache=not args.no_cache,
            include_shots=args.include_shots,
        )

        logger.info("Scraping completed successfully!")
        for name, path in output_files.items():
            logger.info(f"  {name}: {path}")

        return 0

    except Exception as e:
        logger.exception(f"Scraping failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
