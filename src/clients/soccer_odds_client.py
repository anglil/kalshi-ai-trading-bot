"""
Soccer Odds Client

Fetches soccer match data and win probabilities from multiple sources:
  1. ESPN Soccer API — FREE, no auth (multiple leagues)
  2. Simple Elo model from team records — computed locally
  3. The Odds API — optional, requires ODDS_API_KEY
  4. API-Football — optional, requires API_FOOTBALL_KEY

All sources except The Odds API and API-Football require no API keys.
"""

import asyncio
import aiohttp
import math
import ssl
import time
from typing import Dict, List, Optional, Tuple

from src.strategies.sports_common import (
    SportsMultiSourceForecast,
    SportsForecastSource,
    normalize_team_name,
    american_to_prob,
    normalize_probs,
)
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("soccer_odds_client")

# In-memory cache
_cache: Dict[str, Tuple[float, List[SportsMultiSourceForecast]]] = {}
_CACHE_TTL = 600  # 10 minutes

# League configuration
# Maps internal key -> ESPN league code, Kalshi series ticker, Odds API key,
# API-Football league ID, Football-Data code, average draw rate
SOCCER_LEAGUES: Dict[str, Dict] = {
    "epl": {
        "espn_code": "eng.1",
        "kalshi_series": "KXEPLGAME",
        "odds_api_key": "soccer_epl",
        "api_football_id": 39,
        "football_data_code": "PL",
        "draw_rate": 0.24,
        "name": "Premier League",
    },
    "la_liga": {
        "espn_code": "esp.1",
        "kalshi_series": "KXLALIGAGAME",
        "odds_api_key": "soccer_spain_la_liga",
        "api_football_id": 140,
        "football_data_code": "PD",
        "draw_rate": 0.25,
        "name": "La Liga",
    },
    "serie_a": {
        "espn_code": "ita.1",
        "kalshi_series": "KXSERIEA",
        "odds_api_key": "soccer_italy_serie_a",
        "api_football_id": 135,
        "football_data_code": "SA",
        "draw_rate": 0.26,
        "name": "Serie A",
    },
    "bundesliga": {
        "espn_code": "ger.1",
        "kalshi_series": "KXBUNDESLIGAGAME",
        "odds_api_key": "soccer_germany_bundesliga",
        "api_football_id": 78,
        "football_data_code": "BL1",
        "draw_rate": 0.24,
        "name": "Bundesliga",
    },
    "ligue_1": {
        "espn_code": "fra.1",
        "kalshi_series": "KXLIGUE1GAME",
        "odds_api_key": "soccer_france_ligue_one",
        "api_football_id": 61,
        "football_data_code": "FL1",
        "draw_rate": 0.25,
        "name": "Ligue 1",
    },
    "mls": {
        "espn_code": "usa.1",
        "kalshi_series": "KXMLSGAME",
        "odds_api_key": "soccer_usa_mls",
        "api_football_id": 253,
        "football_data_code": None,
        "draw_rate": 0.23,
        "name": "MLS",
    },
    "ucl": {
        "espn_code": "uefa.champions",
        "kalshi_series": "KXUCLGAME",
        "odds_api_key": "soccer_uefa_champs_league",
        "api_football_id": 2,
        "football_data_code": "CL",
        "draw_rate": 0.22,
        "name": "Champions League",
    },
}

# Elo constants
_ELO_BASE = 1500
_ELO_HCA = 65  # home field advantage (smaller than NBA)
_ELO_SCALE = 400


def _make_ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _win_pct_to_elo(win_pct: float) -> float:
    """Convert win percentage to approximate Elo rating."""
    if win_pct <= 0:
        return _ELO_BASE - 300
    if win_pct >= 1:
        return _ELO_BASE + 300
    return _ELO_BASE + _ELO_SCALE * math.log10(win_pct / (1 - win_pct))


def _compute_soccer_elo(
    home_record: Tuple[int, int, int],  # (wins, draws, losses)
    away_record: Tuple[int, int, int],
    league_draw_rate: float = 0.26,
) -> Tuple[float, float, float]:
    """
    Compute win/draw/loss probabilities from team records using Elo + draw model.

    Returns (home_win_prob, away_win_prob, draw_prob).
    """
    home_w, home_d, home_l = home_record
    away_w, away_d, away_l = away_record

    home_games = home_w + home_d + home_l
    away_games = away_w + away_d + away_l

    if home_games == 0 or away_games == 0:
        return 0.40, 0.30, 0.30

    # Points per game as proxy for strength (3 for win, 1 for draw)
    home_ppg = (home_w * 3 + home_d) / home_games
    away_ppg = (away_w * 3 + away_d) / away_games

    # Normalize PPG to 0-1 range (max PPG = 3)
    home_strength = home_ppg / 3.0
    away_strength = away_ppg / 3.0

    home_elo = _win_pct_to_elo(max(0.01, min(0.99, home_strength))) + _ELO_HCA
    away_elo = _win_pct_to_elo(max(0.01, min(0.99, away_strength)))

    # Expected score (Elo-based)
    expected_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / _ELO_SCALE))

    # Split into home_win, draw, away_win using draw rate model
    # Draw probability is highest when teams are evenly matched
    elo_diff = abs(home_elo - away_elo)
    # Draw probability decreases as skill gap increases
    draw_factor = max(0.5, 1.0 - elo_diff / 600.0)
    draw_prob = league_draw_rate * draw_factor

    # Remaining probability split by Elo expected score
    remaining = 1.0 - draw_prob
    home_win = expected_home * remaining
    away_win = (1.0 - expected_home) * remaining

    return home_win, away_win, draw_prob


async def _fetch_espn_soccer(
    session: aiohttp.ClientSession,
    league_code: str,
    league_name: str,
) -> List[Dict]:
    """
    Fetch today's soccer games from ESPN scoreboard for a specific league.

    Returns list of dicts with game info including team names and records.
    """
    games = []
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league_code}/scoreboard"

    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=15),
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"ESPN soccer ({league_name}) returned {resp.status}")
                return games
            data = await resp.json()

        events = data.get("events", [])
        if events:
            logger.info(f"ESPN soccer ({league_name}): {len(events)} games today")

        for event in events:
            game_id = event.get("id", "")
            competitions = event.get("competitions", [])
            if not competitions:
                continue

            comp = competitions[0]
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue

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

            # Extract records (W-D-L format for soccer)
            home_record = (0, 0, 0)  # wins, draws, losses
            away_record = (0, 0, 0)
            for rec in home_comp.get("records", []):
                if rec.get("type") == "total":
                    summary = rec.get("summary", "0-0-0")
                    parts = summary.split("-")
                    if len(parts) == 3:
                        home_record = (int(parts[0]), int(parts[1]), int(parts[2]))
                    elif len(parts) == 2:
                        # Some leagues show W-L only
                        home_record = (int(parts[0]), 0, int(parts[1]))
                    break
            for rec in away_comp.get("records", []):
                if rec.get("type") == "total":
                    summary = rec.get("summary", "0-0-0")
                    parts = summary.split("-")
                    if len(parts) == 3:
                        away_record = (int(parts[0]), int(parts[1]), int(parts[2]))
                    elif len(parts) == 2:
                        away_record = (int(parts[0]), 0, int(parts[1]))
                    break

            # Extract odds if available
            odds_home_ml = None
            odds_away_ml = None
            odds_draw_ml = None
            odds_list = comp.get("odds", [])
            if odds_list:
                odds_data = odds_list[0]
                home_ml = odds_data.get("homeTeamOdds", {}).get("moneyLine")
                away_ml = odds_data.get("awayTeamOdds", {}).get("moneyLine")
                draw_ml = odds_data.get("drawOdds", {}).get("moneyLine")
                if home_ml is not None:
                    odds_home_ml = int(home_ml)
                if away_ml is not None:
                    odds_away_ml = int(away_ml)
                if draw_ml is not None:
                    odds_draw_ml = int(draw_ml)

            start_time = event.get("date", "")

            games.append({
                "game_id": game_id,
                "home_team": home_team_name,
                "away_team": away_team_name,
                "home_record": home_record,
                "away_record": away_record,
                "odds_home_ml": odds_home_ml,
                "odds_away_ml": odds_away_ml,
                "odds_draw_ml": odds_draw_ml,
                "start_time": start_time,
            })

    except Exception as e:
        logger.warning(f"ESPN soccer ({league_name}) fetch failed: {e}")

    return games


async def _fetch_odds_api_soccer(
    session: aiohttp.ClientSession,
    odds_api_sport_key: str,
    api_key: str,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Fetch soccer h2h odds from The Odds API.
    Returns dict mapping "home vs away" -> (home_prob, draw_prob, away_prob).
    """
    if not api_key:
        return {}

    url = (
        f"https://api.the-odds-api.com/v4/sports/{odds_api_sport_key}/odds/"
        f"?apiKey={api_key}&regions=eu&markets=h2h&oddsFormat=decimal"
    )

    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=15),
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"Odds API soccer ({odds_api_sport_key}) returned {resp.status}")
                return {}
            data = await resp.json()

        results = {}
        for game in data:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            bookmakers = game.get("bookmakers", [])

            if not bookmakers:
                continue

            home_probs = []
            draw_probs = []
            away_probs = []

            for bm in bookmakers:
                for market in bm.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes", []):
                        price = outcome.get("price", 0)
                        name = outcome.get("name", "")
                        if price <= 1.0:
                            continue
                        prob = 1.0 / price  # decimal to prob
                        if name == home:
                            home_probs.append(prob)
                        elif name == away:
                            away_probs.append(prob)
                        elif name == "Draw":
                            draw_probs.append(prob)

            if home_probs and away_probs and draw_probs:
                avg_h = sum(home_probs) / len(home_probs)
                avg_a = sum(away_probs) / len(away_probs)
                avg_d = sum(draw_probs) / len(draw_probs)
                normed = normalize_probs([avg_h, avg_a, avg_d])
                key = f"{home} vs {away}"
                results[key] = (normed[0], normed[2], normed[1])  # home, away, draw

        logger.info(f"Odds API soccer ({odds_api_sport_key}): {len(results)} games")
        return results

    except Exception as e:
        logger.warning(f"Odds API soccer fetch failed: {e}")
        return {}


async def _fetch_api_football(
    session: aiohttp.ClientSession,
    api_key: str,
    league_id: int,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Fetch predictions from API-Football.
    Returns dict mapping "home vs away" -> (home_prob, away_prob, draw_prob).
    """
    if not api_key:
        return {}

    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"https://v3.football.api-sports.io/predictions?date={today}&league={league_id}"

    try:
        headers = {"x-apisports-key": api_key}
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=15),
            headers=headers,
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"API-Football returned {resp.status}")
                return {}
            data = await resp.json()

        results = {}
        response = data.get("response", [])
        for pred in response:
            predictions = pred.get("predictions", {})
            percent = predictions.get("percent", {})
            teams = pred.get("teams", {})

            home_name = teams.get("home", {}).get("name", "")
            away_name = teams.get("away", {}).get("name", "")

            home_pct = percent.get("home", "0%").replace("%", "")
            draw_pct = percent.get("draw", "0%").replace("%", "")
            away_pct = percent.get("away", "0%").replace("%", "")

            try:
                h = float(home_pct) / 100.0
                d = float(draw_pct) / 100.0
                a = float(away_pct) / 100.0
                key = f"{home_name} vs {away_name}"
                results[key] = (h, a, d)
            except ValueError:
                continue

        logger.info(f"API-Football league {league_id}: {len(results)} predictions")
        return results

    except Exception as e:
        logger.warning(f"API-Football fetch failed: {e}")
        return {}


async def fetch_soccer_forecasts(
    odds_api_key: str = "",
    api_football_key: str = "",
) -> List[SportsMultiSourceForecast]:
    """
    Main entry point: fetch soccer forecasts from all available sources across leagues.

    Always fetches ESPN + computes Elo. Optionally fetches Odds API / API-Football.
    Returns list of SportsMultiSourceForecast, one per game.
    """
    cache_key = "soccer_forecasts"
    if cache_key in _cache:
        ts, cached = _cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return cached

    all_forecasts = []

    async with aiohttp.ClientSession() as session:
        for league_key, league_config in SOCCER_LEAGUES.items():
            espn_code = league_config["espn_code"]
            league_name = league_config["name"]
            draw_rate = league_config["draw_rate"]

            # Fetch ESPN games for this league
            espn_games = await _fetch_espn_soccer(session, espn_code, league_name)
            if not espn_games:
                continue

            # Fetch Odds API data for this league (if key available)
            odds_data = {}
            if odds_api_key:
                odds_data = await _fetch_odds_api_soccer(
                    session, league_config["odds_api_key"], odds_api_key,
                )

            # Fetch API-Football data for this league (if key available)
            apifb_data = {}
            if api_football_key:
                apifb_data = await _fetch_api_football(
                    session, api_football_key, league_config["api_football_id"],
                )

            for game in espn_games:
                home_code = normalize_team_name(game["home_team"], "soccer")
                away_code = normalize_team_name(game["away_team"], "soccer")

                forecast = SportsMultiSourceForecast(
                    game_id=game["game_id"],
                    home_team=home_code,
                    away_team=away_code,
                    league=league_key,
                    start_time=game.get("start_time"),
                )

                # Source 1: ESPN odds (moneyline if available)
                home_ml = game.get("odds_home_ml")
                away_ml = game.get("odds_away_ml")
                draw_ml = game.get("odds_draw_ml")
                if home_ml is not None and away_ml is not None:
                    probs = [american_to_prob(home_ml), american_to_prob(away_ml)]
                    if draw_ml is not None:
                        probs.append(american_to_prob(draw_ml))
                    normed = normalize_probs(probs)
                    if len(normed) == 3:
                        forecast.sources.append(SportsForecastSource(
                            provider="espn_odds",
                            home_win=normed[0],
                            away_win=normed[1],
                            draw=normed[2],
                        ))
                    else:
                        # No draw odds — estimate draw from league average
                        remaining = 1.0 - draw_rate
                        forecast.sources.append(SportsForecastSource(
                            provider="espn_odds",
                            home_win=normed[0] * remaining,
                            away_win=normed[1] * remaining,
                            draw=draw_rate,
                        ))
                else:
                    forecast.failed_sources.append("espn_odds")

                # Source 2: Elo model from records
                home_record = game.get("home_record", (0, 0, 0))
                away_record = game.get("away_record", (0, 0, 0))
                home_games = sum(home_record)
                away_games = sum(away_record)
                if home_games > 0 and away_games > 0:
                    elo_h, elo_a, elo_d = _compute_soccer_elo(
                        home_record, away_record, draw_rate,
                    )
                    forecast.sources.append(SportsForecastSource(
                        provider="elo",
                        home_win=elo_h,
                        away_win=elo_a,
                        draw=elo_d,
                    ))
                else:
                    forecast.failed_sources.append("elo")

                # Source 3: Odds API (if available)
                for key, (h_prob, a_prob, d_prob) in odds_data.items():
                    if game["home_team"] in key and game["away_team"] in key:
                        forecast.sources.append(SportsForecastSource(
                            provider="odds_api",
                            home_win=h_prob,
                            away_win=a_prob,
                            draw=d_prob,
                        ))
                        break

                # Source 4: API-Football (if available)
                for key, (h_prob, a_prob, d_prob) in apifb_data.items():
                    if game["home_team"] in key and game["away_team"] in key:
                        forecast.sources.append(SportsForecastSource(
                            provider="api_football",
                            home_win=h_prob,
                            away_win=a_prob,
                            draw=d_prob,
                        ))
                        break

                if forecast.sources:
                    all_forecasts.append(forecast)
                    src_names = [s.provider for s in forecast.sources]
                    logger.info(
                        f"Soccer {league_name} {home_code} vs {away_code}: "
                        f"{len(forecast.sources)} sources ({', '.join(src_names)})"
                    )

            # Small delay between leagues
            await asyncio.sleep(0.5)

    _cache[cache_key] = (time.time(), all_forecasts)
    logger.info(f"Soccer forecasts: {len(all_forecasts)} games across all leagues")
    return all_forecasts
