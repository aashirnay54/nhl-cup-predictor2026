"""
CLI entry point for feature engineering pipeline.

Usage:
    python -m src.features
    python -m src.features --raw-data data/raw --output data/processed
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

from .engineering import run_feature_pipeline


def main():
    parser = argparse.ArgumentParser(description="NHL feature engineering pipeline")

    parser.add_argument(
        "--raw-data",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw parquet files (default: data/raw)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed features (default: data/processed)"
    )

    parser.add_argument(
        "-o", "--output-filename",
        type=str,
        default="games_with_features.parquet",
        help="Output filename (default: games_with_features.parquet)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=log_level)

    logger.info("Starting feature engineering pipeline")
    logger.info(f"Raw data: {args.raw_data}")
    logger.info(f"Output: {args.output / args.output_filename}")

    try:
        df = run_feature_pipeline(
            raw_data_dir=args.raw_data,
            output_dir=args.output,
            output_filename=args.output_filename
        )

        logger.success(f"Feature engineering complete: {len(df)} games, {len(df.columns)} features")

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


if __name__ == "__main__":
    main()
