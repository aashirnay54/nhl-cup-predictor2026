"""Tests for data scraping modules."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.scrape.nhl_api import NHLAPIScraper
from src.scrape.moneypuck import MoneyPuckScraper
from src.scrape.pipeline import parse_season_range


class TestParseSeasonRange:
    """Tests for season range parsing."""

    def test_single_season(self):
        assert parse_season_range("2024") == [2024]

    def test_multi_season_range(self):
        assert parse_season_range("2015-2025") == list(range(2015, 2026))

    def test_two_season_range(self):
        assert parse_season_range("2024-2025") == [2024, 2025]

    def test_short_format(self):
        # "2024-25" should convert to [2024, 2025]
        result = parse_season_range("2024-25")
        assert result == [2024, 2025]


class TestNHLAPIScraper:
    """Tests for NHL API scraper."""

    @pytest.fixture
    def scraper(self, tmp_path):
        """Create a scraper with temp cache directory."""
        return NHLAPIScraper(cache_dir=tmp_path, rate_limit=0, use_cache=True)

    @pytest.fixture
    def sample_schedule_response(self):
        """Sample schedule API response."""
        return {
            "gameWeek": [
                {
                    "date": "2024-01-15",
                    "dayAbbrev": "MON",
                    "numberOfGames": 1,
                    "games": [
                        {
                            "id": 2023020672,
                            "season": 20232024,
                            "gameType": 2,
                            "gameState": "OFF",
                            "startTimeUTC": "2024-01-15T17:00:00Z",
                            "homeTeam": {
                                "id": 7,
                                "abbrev": "BUF",
                                "score": 3,
                            },
                            "awayTeam": {
                                "id": 28,
                                "abbrev": "SJS",
                                "score": 0,
                            },
                        }
                    ],
                }
            ]
        }

    @pytest.fixture
    def sample_boxscore_response(self):
        """Sample boxscore API response."""
        return {
            "homeTeam": {"sog": 32},
            "awayTeam": {"sog": 25},
            "playerByGameStats": {
                "homeTeam": {
                    "goalies": [
                        {
                            "playerId": 8480045,
                            "saves": 25,
                            "shotsAgainst": 25,
                        }
                    ]
                },
                "awayTeam": {
                    "goalies": [
                        {
                            "playerId": 8478492,
                            "saves": 29,
                            "shotsAgainst": 32,
                        }
                    ]
                },
            },
        }

    def test_cache_hit(self, scraper, sample_schedule_response):
        """Test that cached responses are used."""
        cache_path = scraper._cache_path("/schedule/2024-01-15")
        cache_path.write_text(json.dumps(sample_schedule_response))

        result = scraper.get_schedule("2024-01-15")

        assert result == sample_schedule_response

    @patch("requests.Session.get")
    def test_rate_limiting(self, mock_get, tmp_path):
        """Test that rate limiting is applied."""
        scraper = NHLAPIScraper(cache_dir=tmp_path, rate_limit=0.1, use_cache=False)

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        import time

        start = time.time()
        scraper._get("/test1")
        scraper._get("/test2")
        elapsed = time.time() - start

        assert elapsed >= 0.1

    def test_parse_boxscore(self, scraper, sample_boxscore_response):
        """Test boxscore parsing."""
        game = {
            "id": 2023020672,
            "startTimeUTC": "2024-01-15T17:00:00Z",
            "season": 20232024,
            "gameType": 2,
            "homeTeam": {"id": 7, "abbrev": "BUF", "score": 3},
            "awayTeam": {"id": 28, "abbrev": "SJS", "score": 0},
        }

        result = scraper._parse_boxscore(game, sample_boxscore_response)

        assert result["game_id"] == 2023020672
        assert result["home_team"] == "BUF"
        assert result["away_team"] == "SJS"
        assert result["home_goals"] == 3
        assert result["away_goals"] == 0
        assert result["home_win"] == 1
        assert result["home_sog"] == 32
        assert result["away_sog"] == 25
        assert result["is_playoff"] is False

    def test_clear_cache(self, scraper, sample_schedule_response):
        """Test cache clearing."""
        cache_path = scraper._cache_path("/test")
        cache_path.write_text(json.dumps(sample_schedule_response))

        assert cache_path.exists()

        scraper.clear_cache()

        assert not cache_path.exists()


class TestMoneyPuckScraper:
    """Tests for MoneyPuck scraper."""

    @pytest.fixture
    def scraper(self, tmp_path):
        """Create a scraper with temp cache directory."""
        return MoneyPuckScraper(cache_dir=tmp_path, rate_limit=0, use_cache=True)

    @pytest.fixture
    def sample_team_csv(self):
        """Sample team stats CSV content."""
        return """team,situation,xGoalsPercentage,corsiPercentage,fenwickPercentage
BUF,all,52.5,51.2,50.8
L.A,all,48.2,49.5,49.1
N.J,all,55.1,54.2,53.8"""

    def test_team_abbrev_mapping(self, scraper, tmp_path, sample_team_csv):
        """Test that team abbreviations are standardized."""
        # Create a cached parquet file
        df = pd.read_csv(pd.io.common.StringIO(sample_team_csv))
        cache_path = tmp_path / "teams_2024_regular.parquet"
        df.to_parquet(cache_path)

        result = scraper.get_team_season_stats(2024)

        assert "LAK" in result["team"].values
        assert "NJD" in result["team"].values
        assert "L.A" not in result["team"].values

    def test_cache_as_parquet(self, scraper, tmp_path, sample_team_csv):
        """Test that data is cached as parquet."""
        with patch("requests.Session.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = sample_team_csv
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            scraper.get_team_season_stats(2024)

            cache_path = tmp_path / "teams_2024_regular.parquet"
            assert cache_path.exists()

            # Verify parquet can be read
            df = pd.read_parquet(cache_path)
            assert len(df) == 3


class TestIntegration:
    """Integration tests (require network access, marked slow)."""

    @pytest.mark.slow
    def test_nhl_api_schedule_fetch(self, tmp_path):
        """Test actual NHL API schedule fetch."""
        scraper = NHLAPIScraper(cache_dir=tmp_path, use_cache=True)
        result = scraper.get_schedule("2024-01-15")

        assert "gameWeek" in result
        assert len(result["gameWeek"]) > 0

    @pytest.mark.slow
    def test_nhl_api_standings_fetch(self, tmp_path):
        """Test actual NHL API standings fetch."""
        scraper = NHLAPIScraper(cache_dir=tmp_path, use_cache=True)
        result = scraper.get_standings("2024-01-15")

        assert "standings" in result

    @pytest.mark.slow
    def test_moneypuck_team_stats_fetch(self, tmp_path):
        """Test actual MoneyPuck team stats fetch."""
        scraper = MoneyPuckScraper(cache_dir=tmp_path, use_cache=True)
        result = scraper.get_team_season_stats(2024)

        assert len(result) > 0
        assert "team" in result.columns
        assert "xGoalsPercentage" in result.columns


# Configure pytest markers
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (requiring network access)"
    )
