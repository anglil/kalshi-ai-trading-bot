"""
Soccer Consensus Trading Strategy

Uses multi-source soccer win probability consensus (ESPN + Elo, optionally
The Odds API and API-Football) to trade Kalshi soccer match winner markets.

Handles 3 outcomes (home/draw/away) across multiple leagues:
EPL, La Liga, Serie A, Bundesliga, Ligue 1, MLS, Champions League.

Uses wider tolerance (0.15 spread threshold mapped to 0.12/0.22 in consensus)
because 2-source consensus has higher variance and draws add uncertainty.
"""

from datetime import datetime
from typing import Dict, List
from zoneinfo import ZoneInfo

from src.clients.kalshi_client import KalshiClient
from src.clients.soccer_odds_client import fetch_soccer_forecasts, SOCCER_LEAGUES
from src.config.settings import settings
from src.paper.tracker import log_signal, get_connection as get_paper_db
from src.strategies.sports_common import (
    SportsMarketOutcome,
    compute_sports_consensus,
    generate_sports_signals,
    execute_sports_trade,
    match_game_to_kalshi_tickers,
)
from src.utils.database import DatabaseManager
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("soccer_consensus")


# ============================================================
# Paper -> Live auto-switch
# ============================================================

def _check_paper_performance(strategy: str) -> bool:
    """
    Check paper trading performance for auto-switch to live.
    Returns True if criteria met: >=20 settled, win_rate>=55%, total_pnl>0.
    """
    try:
        conn = get_paper_db()
        rows = conn.execute(
            "SELECT outcome, pnl FROM signals WHERE strategy = ? AND outcome != 'pending'",
            (strategy,),
        ).fetchall()
        conn.close()

        if len(rows) < 20:
            return False

        wins = sum(1 for r in rows if r["outcome"] == "win")
        win_rate = wins / len(rows) * 100
        total_pnl = sum(r["pnl"] for r in rows if r["pnl"] is not None)

        if win_rate >= 55 and total_pnl > 0:
            logger.info(
                f"SOCCER auto-switch: {len(rows)} settled, "
                f"win_rate={win_rate:.1f}%, pnl=${total_pnl:.2f} — switching to LIVE"
            )
            return True
        return False
    except Exception:
        return False


# ============================================================
# Market discovery (per league)
# ============================================================

async def _discover_soccer_markets(
    kalshi_client: KalshiClient,
    series_ticker: str,
) -> List[Dict]:
    """
    Discover active soccer game markets on Kalshi for a specific league series.
    Returns raw market dicts for ticker matching.
    """
    try:
        markets_response = await kalshi_client.get_markets(
            limit=200,
            series_ticker=series_ticker,
            status="open",
        )
        markets = markets_response.get("markets", [])

        active = [
            m for m in markets
            if m.get("status") in ("active", "open")
        ]

        if active:
            logger.info(
                f"Soccer markets ({series_ticker}): "
                f"{len(active)} active markets discovered"
            )

        return active

    except Exception as e:
        logger.debug(f"Soccer market discovery ({series_ticker}) failed: {e}")
        return []


# ============================================================
# Main soccer consensus trading cycle
# ============================================================

async def run_soccer_consensus_cycle(
    kalshi_client: KalshiClient,
    db_manager: DatabaseManager,
    paper_mode: bool = True,
) -> Dict:
    """
    Run one complete soccer consensus trading cycle.

    1. Auto-switch check
    2. Fetch forecasts (ESPN + Elo, optionally Odds API / API-Football)
    3. For each league: discover Kalshi markets
    4. For each game: match tickers -> compute consensus -> generate signals
    5. Sort by edge, execute top 6 signals (paper or live)
    """
    strategy_tag = "soccer_consensus"
    logger.info(f"SOCCER CONSENSUS: Starting cycle (paper={paper_mode})...")

    # Auto-switch check
    if paper_mode and _check_paper_performance(strategy_tag):
        paper_mode = False
        logger.info("SOCCER CONSENSUS: Auto-switched to LIVE mode!")

    results: Dict = {
        "leagues_scanned": 0,
        "markets_found": 0,
        "games_analyzed": 0,
        "signals_generated": 0,
        "orders_placed": 0,
        "total_position_value": 0.0,
        "paper_mode": paper_mode,
    }

    # 1. Fetch forecasts across all leagues
    odds_api_key = settings.api.odds_api_key
    api_football_key = settings.api.api_football_key
    forecasts = await fetch_soccer_forecasts(
        odds_api_key=odds_api_key,
        api_football_key=api_football_key,
    )
    if not forecasts:
        logger.warning("SOCCER CONSENSUS: No soccer forecasts available")
        return results

    # 2. Discover Kalshi markets per league
    # Group forecasts by league, then discover markets for each league
    leagues_with_games = set(f.league for f in forecasts)
    kalshi_markets_by_league: Dict[str, List[Dict]] = {}

    for league_key in leagues_with_games:
        league_config = SOCCER_LEAGUES.get(league_key)
        if not league_config:
            continue

        series_ticker = league_config["kalshi_series"]
        markets = await _discover_soccer_markets(kalshi_client, series_ticker)
        results["leagues_scanned"] += 1
        results["markets_found"] += len(markets)
        if markets:
            kalshi_markets_by_league[league_key] = markets

    if not kalshi_markets_by_league:
        logger.info("SOCCER CONSENSUS: No active soccer markets found on Kalshi")
        return results

    # 3. Get bankroll
    if paper_mode:
        bankroll = 1000.0
    else:
        try:
            bankroll = await kalshi_client.get_total_portfolio_value()
        except Exception as e:
            logger.error(f"Could not fetch portfolio value: {e}")
            return results
        if bankroll < 5.0:
            logger.warning(f"SOCCER CONSENSUS: Insufficient bankroll: ${bankroll:.2f}")
            return results

    # 4. Process each game
    all_signals = []

    for forecast in forecasts:
        results["games_analyzed"] += 1
        league_key = forecast.league

        # Get Kalshi markets for this league
        league_markets = kalshi_markets_by_league.get(league_key, [])
        if not league_markets:
            continue

        # Compute consensus
        consensus = compute_sports_consensus(forecast, min_sources=2)
        if consensus is None:
            logger.debug(
                f"Soccer {forecast.home_team} vs {forecast.away_team}: "
                f"insufficient sources ({len(forecast.sources)})"
            )
            continue

        if consensus.confidence == "low":
            logger.info(
                f"Soccer {forecast.home_team} vs {forecast.away_team}: "
                f"low confidence (spread={consensus.max_spread:.2f}), skipping"
            )
            continue

        league_name = SOCCER_LEAGUES.get(league_key, {}).get("name", league_key)
        logger.info(
            f"Soccer {league_name} {forecast.home_team} vs {forecast.away_team}: "
            f"consensus H={consensus.home_prob:.0%} D={consensus.draw_prob:.0%} "
            f"A={consensus.away_prob:.0%} confidence={consensus.confidence} "
            f"(spread={consensus.max_spread:.2f}, sources={consensus.source_count})"
        )

        # Match to Kalshi tickers (soccer has 3 outcomes: home/draw/away)
        outcomes = match_game_to_kalshi_tickers(
            forecast.home_team, forecast.away_team,
            league_markets, has_draw=True,
        )

        if not outcomes:
            logger.debug(
                f"Soccer {forecast.home_team} vs {forecast.away_team}: "
                f"no matching Kalshi markets"
            )
            continue

        # Generate signals
        game_desc = f"{league_name}: {forecast.home_team} vs {forecast.away_team}"
        signals = generate_sports_signals(
            outcomes=outcomes,
            consensus=consensus,
            game_desc=game_desc,
            bankroll=bankroll,
            min_edge=0.08,
            max_position_pct=0.05,
            kelly_fraction=0.5,
            rationale_prefix=f"SOCCER({consensus.confidence})",
        )

        all_signals.extend(signals)

    results["signals_generated"] = len(all_signals)

    if not all_signals:
        logger.info("SOCCER CONSENSUS: No trade signals generated")
        return results

    # 5. Sort by edge and execute top signals
    all_signals.sort(key=lambda s: s.edge, reverse=True)
    max_trades = 6  # More trades than NBA since multiple leagues

    logger.info(f"SOCCER CONSENSUS: Top signals ({len(all_signals)} total):")
    for i, sig in enumerate(all_signals[:max_trades]):
        logger.info(f"  #{i+1}: {sig.rationale}")

    for signal in all_signals[:max_trades]:
        if paper_mode:
            log_signal(
                market_id=signal.market.ticker,
                market_title=signal.rationale,
                side=signal.side,
                entry_price=signal.limit_price / 100.0,
                confidence=signal.confidence,
                reasoning=signal.rationale,
                strategy=strategy_tag,
            )
            results["orders_placed"] += 1
            logger.info(
                f"SOCCER PAPER TRADE: {signal.market.ticker} {signal.side} "
                f"@ {signal.limit_price}c"
            )
        else:
            success = await execute_sports_trade(
                signal, kalshi_client, db_manager,
                strategy=strategy_tag,
            )
            if success:
                results["orders_placed"] += 1
                results["total_position_value"] += signal.position_size_dollars

    logger.info(
        f"SOCCER CONSENSUS cycle complete: {results['leagues_scanned']} leagues, "
        f"{results['games_analyzed']} games, {results['signals_generated']} signals, "
        f"{results['orders_placed']} orders ({'PAPER' if paper_mode else 'LIVE'})"
    )
    return results
