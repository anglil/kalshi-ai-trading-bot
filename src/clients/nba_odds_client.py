"""
NBA Odds Client

Fetches NBA game data and win probabilities from multiple sources:
  1. ESPN BPI (Matchup Predictor) — FREE, no auth
  2. Simple Elo model from team records — computed locally
  3. The Odds API — optional, requires ODDS_API_KEY

All sources except The Odds API require no API keys.
"""

import asyncio
import aiohttp
import math
import ssl
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.strategies.sports_common import (
    SportsMultiSourceForecast,
    SportsForecastSource,
    normalize_team_name,
)
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("nba_odds_client")

# In-memory cache
_cache: Dict[str, Tuple[float, List[SportsMultiSourceForecast]]] = {}
_CACHE_TTL = 600  # 10 minutes

# Elo constants
_ELO_BASE = 1500
_ELO_HCA = 100  # home court advantage in Elo points
_ELO_SCALE = 400  # standard Elo scale factor


def _make_ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _win_pct_to_elo(win_pct: float) -> float:
    """Convert a team's win percentage to approximate Elo rating."""
    if win_pct <= 0:
        return _ELO_BASE - 300
    if win_pct >= 1:
        return _ELO_BASE + 300
    # Logistic approximation: Elo ~ base + scale * log10(w / (1-w))
    return _ELO_BASE + _ELO_SCALE * math.log10(win_pct / (1 - win_pct))


def _compute_elo_prob(
    home_record: Tuple[int, int],
    away_record: Tuple[int, int],
) -> Tuple[float, float]:
    """
    Compute win probabilities from team records using a simple Elo model.

    Returns (home_win_prob, away_win_prob).
    """
    home_wins, home_losses = home_record
    away_wins, away_losses = away_record

    home_games = home_wins + home_losses
    away_games = away_wins + away_losses

    if home_games == 0 or away_games == 0:
        return 0.5, 0.5

    home_pct = home_wins / home_games
    away_pct = away_wins / away_games

    home_elo = _win_pct_to_elo(home_pct) + _ELO_HCA
    away_elo = _win_pct_to_elo(away_pct)

    expected_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / _ELO_SCALE))
    expected_away = 1.0 - expected_home

    return expected_home, expected_away


async def _fetch_espn_nba(
    session: aiohttp.ClientSession,
) -> List[Dict]:
    """
    Fetch today's NBA games from ESPN scoreboard + BPI matchup predictor.

    Returns list of dicts with keys:
      game_id, home_team, away_team, home_record, away_record,
      bpi_home_win, bpi_away_win, odds_home_ml, odds_away_ml
    """
    games = []
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=15),
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"ESPN NBA scoreboard returned {resp.status}")
                return games
            data = await resp.json()

        events = data.get("events", [])
        logger.info(f"ESPN NBA: {len(events)} games on today's scoreboard")

        for event in events:
            game_id = event.get("id", "")
            competitions = event.get("competitions", [])
            if not competitions:
                continue

            comp = competitions[0]
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue

            # ESPN lists home first, away second (homeAway field confirms)
            home_comp = None
            away_comp = None
            for c in competitors:
                if c.get("homeAway") == "home":
                    home_comp = c
                elif c.get("homeAway") == "away":
                    away_comp = c

            if not home_comp or not away_comp:
                continue

            home_team_name = home_comp.get("team", {}).get("displayName", "")
            away_team_name = away_comp.get("team", {}).get("displayName", "")

            # Extract records
            home_record = (0, 0)
            away_record = (0, 0)
            for rec in home_comp.get("records", []):
                if rec.get("type") == "total":
                    summary = rec.get("summary", "0-0")
                    parts = summary.split("-")
                    if len(parts) == 2:
                        home_record = (int(parts[0]), int(parts[1]))
                    break
            for rec in away_comp.get("records", []):
                if rec.get("type") == "total":
                    summary = rec.get("summary", "0-0")
                    parts = summary.split("-")
                    if len(parts) == 2:
                        away_record = (int(parts[0]), int(parts[1]))
                    break

            # Extract DraftKings odds if available
            odds_home_ml = None
            odds_away_ml = None
            odds_list = comp.get("odds", [])
            if odds_list:
                odds_data = odds_list[0]
                home_ml = odds_data.get("homeTeamOdds", {}).get("moneyLine")
                away_ml = odds_data.get("awayTeamOdds", {}).get("moneyLine")
                if home_ml is not None:
                    odds_home_ml = int(home_ml)
                if away_ml is not None:
                    odds_away_ml = int(away_ml)

            start_time = event.get("date", "")

            game_info = {
                "game_id": game_id,
                "home_team": home_team_name,
                "away_team": away_team_name,
                "home_record": home_record,
                "away_record": away_record,
                "odds_home_ml": odds_home_ml,
                "odds_away_ml": odds_away_ml,
                "start_time": start_time,
            }

            # Try to get BPI matchup predictor from summary endpoint
            bpi_home, bpi_away = await _fetch_bpi_predictor(session, game_id)
            game_info["bpi_home_win"] = bpi_home
            game_info["bpi_away_win"] = bpi_away

            games.append(game_info)

            # Small delay between summary requests to be polite
            await asyncio.sleep(0.3)

    except Exception as e:
        logger.warning(f"ESPN NBA scoreboard fetch failed: {e}")

    return games


async def _fetch_bpi_predictor(
    session: aiohttp.ClientSession,
    game_id: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Fetch BPI matchup predictor for a specific game.
    Returns (home_win_prob, away_win_prob) or (None, None).
    """
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"

    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=10),
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                return None, None
            data = await resp.json()

        # BPI predictor is under "predictor" key
        predictor = data.get("predictor", {})
        if not predictor:
            # Also check "winprobability" or "gameProjection"
            header = data.get("header", {})
            competitions = header.get("competitions", [])
            if competitions:
                comp = competitions[0]
                for c in comp.get("competitors", []):
                    # Some endpoints have gameProjection
                    proj = c.get("gameProjection")
                    if proj is not None:
                        pass  # Handle below

        home_prob = predictor.get("homeTeam", {}).get("gameProjection")
        away_prob = predictor.get("awayTeam", {}).get("gameProjection")

        if home_prob is not None and away_prob is not None:
            # ESPN returns as percentage (e.g., 71.5)
            return float(home_prob) / 100.0, float(away_prob) / 100.0

        return None, None

    except Exception as e:
        logger.debug(f"BPI predictor fetch failed for game {game_id}: {e}")
        return None, None


async def _fetch_odds_api_nba(
    session: aiohttp.ClientSession,
    api_key: str,
) -> Dict[str, Tuple[float, float]]:
    """
    Fetch NBA h2h odds from The Odds API.
    Returns dict mapping "home_team vs away_team" -> (home_prob, away_prob).
    Skipped if api_key is empty.
    """
    if not api_key:
        return {}

    url = (
        f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
        f"?apiKey={api_key}&regions=us&markets=h2h&oddsFormat=american"
    )

    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=15),
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"Odds API returned {resp.status}")
                return {}
            data = await resp.json()

        results = {}
        for game in data:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            bookmakers = game.get("bookmakers", [])

            if not bookmakers:
                continue

            # Average across bookmakers
            home_probs = []
            away_probs = []
            for bm in bookmakers:
                for market in bm.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes", []):
                        price = outcome.get("price", 0)
                        if outcome.get("name") == home and price != 0:
                            from src.strategies.sports_common import american_to_prob
                            home_probs.append(american_to_prob(price))
                        elif outcome.get("name") == away and price != 0:
                            from src.strategies.sports_common import american_to_prob
                            away_probs.append(american_to_prob(price))

            if home_probs and away_probs:
                avg_home = sum(home_probs) / len(home_probs)
                avg_away = sum(away_probs) / len(away_probs)
                # Normalize
                total = avg_home + avg_away
                if total > 0:
                    avg_home /= total
                    avg_away /= total
                key = f"{home} vs {away}"
                results[key] = (avg_home, avg_away)

        logger.info(f"Odds API: {len(results)} NBA games with h2h odds")
        return results

    except Exception as e:
        logger.warning(f"Odds API NBA fetch failed: {e}")
        return {}


async def fetch_nba_forecasts(
    odds_api_key: str = "",
) -> List[SportsMultiSourceForecast]:
    """
    Main entry point: fetch NBA forecasts from all available sources.

    Always fetches ESPN + computes Elo. Optionally fetches Odds API if key provided.
    Returns list of SportsMultiSourceForecast, one per game.
    """
    cache_key = "nba_forecasts"
    if cache_key in _cache:
        ts, cached = _cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return cached

    forecasts = []

    async with aiohttp.ClientSession() as session:
        # Fetch ESPN games (includes BPI + records)
        espn_games = await _fetch_espn_nba(session)

        # Fetch Odds API if key available
        odds_api_data = await _fetch_odds_api_nba(session, odds_api_key)

    for game in espn_games:
        home_code = normalize_team_name(game["home_team"], "nba")
        away_code = normalize_team_name(game["away_team"], "nba")

        forecast = SportsMultiSourceForecast(
            game_id=game["game_id"],
            home_team=home_code,
            away_team=away_code,
            league="nba",
            start_time=game.get("start_time"),
        )

        # Source 1: ESPN BPI
        bpi_home = game.get("bpi_home_win")
        bpi_away = game.get("bpi_away_win")
        if bpi_home is not None and bpi_away is not None:
            forecast.sources.append(SportsForecastSource(
                provider="espn_bpi",
                home_win=bpi_home,
                away_win=bpi_away,
                draw=0.0,
            ))
        else:
            # Fall back to moneyline odds from ESPN/DraftKings
            home_ml = game.get("odds_home_ml")
            away_ml = game.get("odds_away_ml")
            if home_ml is not None and away_ml is not None:
                from src.strategies.sports_common import american_to_prob, normalize_probs
                home_p = american_to_prob(home_ml)
                away_p = american_to_prob(away_ml)
                normed = normalize_probs([home_p, away_p])
                forecast.sources.append(SportsForecastSource(
                    provider="espn_odds",
                    home_win=normed[0],
                    away_win=normed[1],
                    draw=0.0,
                ))
            else:
                forecast.failed_sources.append("espn_bpi")

        # Source 2: Elo model from records
        home_record = game.get("home_record", (0, 0))
        away_record = game.get("away_record", (0, 0))
        if home_record[0] + home_record[1] > 0 and away_record[0] + away_record[1] > 0:
            elo_home, elo_away = _compute_elo_prob(home_record, away_record)
            forecast.sources.append(SportsForecastSource(
                provider="elo",
                home_win=elo_home,
                away_win=elo_away,
                draw=0.0,
            ))
        else:
            forecast.failed_sources.append("elo")

        # Source 3: Odds API (if available)
        # Try to match by team names
        for key, (h_prob, a_prob) in odds_api_data.items():
            # Check if game teams match
            if (game["home_team"] in key and game["away_team"] in key):
                forecast.sources.append(SportsForecastSource(
                    provider="odds_api",
                    home_win=h_prob,
                    away_win=a_prob,
                    draw=0.0,
                ))
                break

        if forecast.sources:
            forecasts.append(forecast)
            src_names = [s.provider for s in forecast.sources]
            logger.info(
                f"NBA {home_code} vs {away_code}: "
                f"{len(forecast.sources)} sources ({', '.join(src_names)})"
            )

    _cache[cache_key] = (time.time(), forecasts)
    logger.info(f"NBA forecasts: {len(forecasts)} games with data")
    return forecasts
