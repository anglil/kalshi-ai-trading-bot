"""
NBA Consensus Trading Strategy

Uses multi-source NBA win probability consensus (ESPN BPI + Elo, optionally
The Odds API) to trade Kalshi NBA game winner markets.

Sports markets are direct probability markets — no Gaussian bracket model.
Consensus = median of win probabilities. Edge = consensus - market price.

NBA games run 6 PM - 1 AM ET, so this strategy uses its own game-hours
check instead of the general night mode.
"""

from datetime import datetime
from typing import Dict, List
from zoneinfo import ZoneInfo

from src.clients.kalshi_client import KalshiClient
from src.clients.nba_odds_client import fetch_nba_forecasts
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

logger = get_trading_logger("nba_consensus")

# Kalshi series ticker for NBA games
NBA_SERIES_TICKER = "KXNBAGAME"


# ============================================================
# NBA game hours check
# ============================================================

def _is_nba_game_hours() -> bool:
    """
    Check if current time is within NBA game hours (6 PM - 1 AM ET).
    NBA games run late, so this is wider than the general night mode window.
    """
    eastern = ZoneInfo("US/Eastern")
    now_et = datetime.now(eastern)
    hour = now_et.hour
    # 6 PM (18) to 1 AM (1): hour >= 18 OR hour < 1
    return hour >= 18 or hour < 1


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
                f"NBA auto-switch: {len(rows)} settled, "
                f"win_rate={win_rate:.1f}%, pnl=${total_pnl:.2f} — switching to LIVE"
            )
            return True
        return False
    except Exception:
        return False


# ============================================================
# Market discovery
# ============================================================

async def _discover_nba_markets(kalshi_client: KalshiClient) -> List[Dict]:
    """
    Discover active NBA game markets on Kalshi.
    Returns raw market dicts for ticker matching.
    """
    try:
        markets_response = await kalshi_client.get_markets(
            limit=200,
            series_ticker=NBA_SERIES_TICKER,
            status="open",
        )
        markets = markets_response.get("markets", [])

        # Filter to active/open markets only
        active = [
            m for m in markets
            if m.get("status") in ("active", "open")
        ]

        if active:
            logger.info(f"NBA markets: {len(active)} active markets discovered")
            for m in active[:3]:
                logger.debug(
                    f"RAW NBA MARKET: ticker={m.get('ticker')}, "
                    f"title={m.get('title')!r}, yes_price={m.get('yes_price')}"
                )

        return active

    except Exception as e:
        logger.warning(f"NBA market discovery failed: {e}")
        return []


# ============================================================
# Main NBA consensus trading cycle
# ============================================================

async def run_nba_consensus_cycle(
    kalshi_client: KalshiClient,
    db_manager: DatabaseManager,
    paper_mode: bool = True,
) -> Dict:
    """
    Run one complete NBA consensus trading cycle.

    1. Auto-switch check
    2. Fetch forecasts (ESPN BPI + Elo, optionally Odds API)
    3. Discover Kalshi NBA markets
    4. For each game: match tickers -> compute consensus -> generate signals
    5. Sort by edge, execute top 4 signals (paper or live)
    """
    strategy_tag = "nba_consensus"
    logger.info(f"NBA CONSENSUS: Starting cycle (paper={paper_mode})...")

    # Auto-switch check
    if paper_mode and _check_paper_performance(strategy_tag):
        paper_mode = False
        logger.info("NBA CONSENSUS: Auto-switched to LIVE mode!")

    results: Dict = {
        "markets_found": 0,
        "games_analyzed": 0,
        "signals_generated": 0,
        "orders_placed": 0,
        "total_position_value": 0.0,
        "paper_mode": paper_mode,
    }

    # 1. Fetch forecasts
    odds_api_key = settings.api.odds_api_key
    forecasts = await fetch_nba_forecasts(odds_api_key=odds_api_key)
    if not forecasts:
        logger.warning("NBA CONSENSUS: No NBA forecasts available")
        return results

    # 2. Discover Kalshi NBA markets
    kalshi_markets = await _discover_nba_markets(kalshi_client)
    results["markets_found"] = len(kalshi_markets)

    if not kalshi_markets:
        logger.info("NBA CONSENSUS: No active NBA markets found on Kalshi")
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
            logger.warning(f"NBA CONSENSUS: Insufficient bankroll: ${bankroll:.2f}")
            return results

    # 4. Process each game
    all_signals = []

    for forecast in forecasts:
        results["games_analyzed"] += 1

        # Compute consensus
        consensus = compute_sports_consensus(forecast, min_sources=2)
        if consensus is None:
            logger.debug(
                f"NBA {forecast.home_team} vs {forecast.away_team}: "
                f"insufficient sources ({len(forecast.sources)})"
            )
            continue

        if consensus.confidence == "low":
            logger.info(
                f"NBA {forecast.home_team} vs {forecast.away_team}: "
                f"low confidence (spread={consensus.max_spread:.2f}), skipping"
            )
            continue

        logger.info(
            f"NBA {forecast.home_team} vs {forecast.away_team}: "
            f"consensus home={consensus.home_prob:.0%} away={consensus.away_prob:.0%} "
            f"confidence={consensus.confidence} (spread={consensus.max_spread:.2f}, "
            f"sources={consensus.source_count})"
        )

        # Match to Kalshi tickers
        outcomes = match_game_to_kalshi_tickers(
            forecast.home_team, forecast.away_team,
            kalshi_markets, has_draw=False,
        )

        if not outcomes:
            logger.debug(
                f"NBA {forecast.home_team} vs {forecast.away_team}: "
                f"no matching Kalshi markets"
            )
            continue

        # Generate signals
        game_desc = f"{forecast.away_team}@{forecast.home_team}"
        signals = generate_sports_signals(
            outcomes=outcomes,
            consensus=consensus,
            game_desc=game_desc,
            bankroll=bankroll,
            min_edge=0.08,
            max_position_pct=0.05,
            kelly_fraction=0.5,
            rationale_prefix=f"NBA({consensus.confidence})",
        )

        all_signals.extend(signals)

    results["signals_generated"] = len(all_signals)

    if not all_signals:
        logger.info("NBA CONSENSUS: No trade signals generated")
        return results

    # 5. Sort by edge and execute top signals
    all_signals.sort(key=lambda s: s.edge, reverse=True)
    max_trades = 4

    logger.info(f"NBA CONSENSUS: Top signals ({len(all_signals)} total):")
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
                f"NBA PAPER TRADE: {signal.market.ticker} {signal.side} "
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
        f"NBA CONSENSUS cycle complete: {results['games_analyzed']} games, "
        f"{results['signals_generated']} signals, {results['orders_placed']} orders "
        f"({'PAPER' if paper_mode else 'LIVE'})"
    )
    return results
