"""
Flu/Respiratory Consensus Trading Strategy

Uses multi-source ILI (influenza-like illness) data consensus from CDC and
Delphi to trade Kalshi flu activity bracket markets. Handles both numeric
ILI% brackets and categorical activity level titles.

Mirrors the weather consensus pattern with flu-specific parameters.
"""

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from src.clients.flu_forecast_client import (
    ILI_THRESHOLDS,
    MultiSourceForecast,
    fetch_flu_forecasts,
    ili_to_level,
    level_to_ili_midpoint,
)
from src.clients.kalshi_client import KalshiClient
from src.strategies.weather_strategy import (
    TemperatureBracket,
    WeatherTradeSignal,
    forecast_to_bracket_probs,
    generate_weather_signals,
    execute_weather_trade,
)
from src.paper.tracker import log_signal, get_connection as get_paper_db
from src.utils.database import DatabaseManager
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("flu_consensus")


# ============================================================
# Consensus parameters for flu/ILI
# ============================================================

FLU_CONSENSUS_PARAMS = {
    "cluster_tolerance": 0.3,    # ILI percentage points
    "sigma_high": 0.2,          # tight sigma when high agreement
    "sigma_med": 0.5,           # medium sigma
    "sigma_low": 0.8,           # wide sigma when low agreement
}

# Series tickers to try on Kalshi for flu markets
FLU_SERIES_CANDIDATES = ["FLU", "KXFLU", "ILI", "FLUACTIVITY", "INFLUENZA", "RESPIRATORY"]

# Categorical level to numeric ILI% mapping for bracket parsing
LEVEL_TO_NUMERIC = {
    "minimal": (0, 200),     # 0% - 2.0% → scale x100 for bracket math
    "low": (200, 260),       # 2.0% - 2.6%
    "moderate": (260, 350),  # 2.6% - 3.5%
    "high": (350, 500),      # 3.5% - 5.0%
    "very high": (500, 800), # 5.0% - 8.0% (capped)
    "very_high": (500, 800),
}


# ============================================================
# Consensus algorithm
# ============================================================

@dataclass
class FluConsensusResult:
    """Result of flu ILI multi-source consensus."""
    consensus_ili: float         # consensus ILI percentage
    consensus_level: str         # "minimal", "low", "moderate", "high", "very_high"
    sigma: float                 # Gaussian sigma in ILI%
    confidence: str              # "high", "medium", "low", "skip"
    agreement_ratio: float
    cluster_values: List[float]
    all_values: List[float]
    source_count: int


def compute_flu_consensus(forecast: MultiSourceForecast) -> FluConsensusResult:
    """
    Determine consensus ILI value and confidence from multi-source data.

    Algorithm mirrors weather consensus with flu-specific tolerances.
    """
    values = [s.value for s in forecast.sources]
    n = len(values)

    if n < 2:
        fallback = values[0] if values else 2.5
        return FluConsensusResult(
            consensus_ili=fallback,
            consensus_level=ili_to_level(fallback),
            sigma=FLU_CONSENSUS_PARAMS["sigma_low"],
            confidence="skip",
            agreement_ratio=0.0,
            cluster_values=values,
            all_values=values,
            source_count=n,
        )

    sorted_values = sorted(values)
    tolerance = FLU_CONSENSUS_PARAMS["cluster_tolerance"]

    # Sliding window: find largest cluster within tolerance
    best_cluster: List[float] = []
    for i in range(n):
        cluster = [sorted_values[i]]
        for j in range(i + 1, n):
            if sorted_values[j] - sorted_values[i] <= tolerance:
                cluster.append(sorted_values[j])
            else:
                break
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    agreement_ratio = len(best_cluster) / n

    if agreement_ratio >= 0.80:
        sigma = FLU_CONSENSUS_PARAMS["sigma_high"]
        confidence = "high"
    elif agreement_ratio >= 0.60:
        sigma = FLU_CONSENSUS_PARAMS["sigma_med"]
        confidence = "medium"
    else:
        sigma = FLU_CONSENSUS_PARAMS["sigma_low"]
        confidence = "low"

    # Consensus = median of agreeing cluster
    best_cluster.sort()
    mid = len(best_cluster) // 2
    if len(best_cluster) % 2 == 0:
        consensus_ili = (best_cluster[mid - 1] + best_cluster[mid]) / 2.0
    else:
        consensus_ili = best_cluster[mid]

    return FluConsensusResult(
        consensus_ili=consensus_ili,
        consensus_level=ili_to_level(consensus_ili),
        sigma=sigma,
        confidence=confidence,
        agreement_ratio=agreement_ratio,
        cluster_values=best_cluster,
        all_values=values,
        source_count=n,
    )


# ============================================================
# Market discovery + bracket parsing
# ============================================================

async def _discover_flu_markets(kalshi_client: KalshiClient) -> List[TemperatureBracket]:
    """
    Discover active flu/ILI bracket markets on Kalshi.
    Tries multiple series ticker candidates.
    """
    brackets = []

    for series in FLU_SERIES_CANDIDATES:
        try:
            markets_response = await kalshi_client.get_markets(
                limit=100,
                series_ticker=series,
            )
            markets = markets_response.get("markets", [])
            if not markets:
                continue

            logger.info(f"Found {len(markets)} markets for series '{series}'")

            for market in markets:
                if market.get("status") != "active":
                    continue
                bracket = _parse_flu_bracket(market)
                if bracket:
                    brackets.append(bracket)

            if brackets:
                break

        except Exception as e:
            logger.debug(f"Series '{series}' lookup failed: {e}")
            continue

    logger.info(f"Flu markets: {len(brackets)} active brackets discovered")
    return brackets


def _parse_flu_bracket(market: dict) -> Optional[TemperatureBracket]:
    """
    Parse a flu bracket from a Kalshi market.

    Handles both numeric ILI% brackets and categorical level titles:
      Numeric: "X%-Y%", ">X%", "<X%"
      Categorical: "minimal", "low", "moderate", "high", "very high"
    """
    title = market.get("title", "").lower()
    ticker = market.get("ticker", "")

    yes_price = market.get("yes_price") or market.get("yes_bid") or market.get("yes_ask")
    no_price = market.get("no_price") or market.get("no_bid") or market.get("no_ask")

    if yes_price is None and no_price is None:
        return None

    if yes_price is not None and no_price is None:
        no_price = 100 - yes_price
    elif no_price is not None and yes_price is None:
        yes_price = 100 - no_price

    yes_ask = market.get("yes_ask") or yes_price
    no_ask = market.get("no_ask") or no_price
    volume = market.get("volume", 0)

    low = None
    high = None

    # First check for categorical level mentions
    for level, (level_low, level_high) in LEVEL_TO_NUMERIC.items():
        if level.replace("_", " ") in title:
            low = level_low
            high = level_high
            break

    # If no categorical match, try numeric ILI% patterns
    if low is None and high is None:
        # Scale: multiply by 100 for basis-point-like bracket math
        range_pct = re.search(r'(\d+\.?\d*)\s*%?\s*[-–to]+\s*(\d+\.?\d*)\s*%', title)
        above_pct = re.search(r'(?:above|>|over|at least)\s*(\d+\.?\d*)\s*%', title)
        below_pct = re.search(r'(?:below|<|under|at most)\s*(\d+\.?\d*)\s*%', title)

        if range_pct:
            low = int(round(float(range_pct.group(1)) * 100))
            high = int(round(float(range_pct.group(2)) * 100))
        elif above_pct and not below_pct:
            low = int(round(float(above_pct.group(1)) * 100))
        elif below_pct and not above_pct:
            high = int(round(float(below_pct.group(1)) * 100))

    if low is None and high is None:
        return None

    return TemperatureBracket(
        ticker=ticker,
        low=low,
        high=high,
        yes_price=yes_price,
        no_price=no_price,
        yes_ask=yes_ask,
        no_ask=no_ask,
        volume=volume,
    )


# ============================================================
# Paper → Live auto-switch
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
                f"FLU auto-switch: {len(rows)} settled, "
                f"win_rate={win_rate:.1f}%, pnl=${total_pnl:.2f} — switching to LIVE"
            )
            return True
        return False
    except Exception:
        return False


# ============================================================
# Main flu consensus trading cycle
# ============================================================

async def run_flu_consensus_cycle(
    kalshi_client: KalshiClient,
    db_manager: DatabaseManager,
    paper_mode: bool = True,
) -> Dict:
    """
    Run one complete flu/ILI consensus trading cycle.

    1. Fetch ILI data from 3 sources
    2. Compute consensus
    3. Discover Kalshi flu markets
    4. Convert consensus to bracket probabilities
    5. Generate trade signals
    6. Execute trades (paper or live)
    """
    strategy_tag = "flu_consensus"
    logger.info(f"FLU CONSENSUS: Starting cycle (paper={paper_mode})...")

    # Auto-switch check
    if paper_mode and _check_paper_performance(strategy_tag):
        paper_mode = False
        logger.info("FLU CONSENSUS: Auto-switched to LIVE mode!")

    results: Dict = {
        "markets_found": 0,
        "brackets_found": 0,
        "signals_generated": 0,
        "orders_placed": 0,
        "total_position_value": 0.0,
        "paper_mode": paper_mode,
    }

    # 1. Fetch flu data
    forecast = await fetch_flu_forecasts()
    if not forecast.sources:
        logger.warning("FLU CONSENSUS: No ILI sources available")
        return results

    # 2. Compute consensus
    consensus = compute_flu_consensus(forecast)

    logger.info(
        f"FLU CONSENSUS: confidence={consensus.confidence}, "
        f"ILI={consensus.consensus_ili:.2f}% (level={consensus.consensus_level}), "
        f"sigma={consensus.sigma:.2f}, agree={consensus.agreement_ratio:.0%} "
        f"(values={[f'{v:.2f}%' for v in consensus.all_values]})"
    )

    if consensus.confidence in ("low", "skip"):
        logger.info("FLU CONSENSUS: skipping (low confidence)")
        return results

    # 3. Discover Kalshi flu markets
    brackets = await _discover_flu_markets(kalshi_client)
    results["brackets_found"] = len(brackets)

    if not brackets:
        logger.info("FLU CONSENSUS: no active flu brackets found")
        return results

    # 4. Calculate bracket probabilities
    # Convert ILI% to the scale used by brackets (x100)
    consensus_scaled = int(round(consensus.consensus_ili * 100))
    sigma_scaled = consensus.sigma * 100

    bracket_probs = forecast_to_bracket_probs(
        consensus_scaled, brackets, sigma=sigma_scaled,
    )

    # 5. Generate trade signals
    try:
        balance_response = await kalshi_client.get_balance()
        bankroll = balance_response.get("balance", 0) / 100.0
    except Exception as e:
        logger.error(f"Could not fetch balance: {e}")
        return results

    if bankroll < 5.0:
        logger.warning(f"FLU CONSENSUS: Insufficient bankroll: ${bankroll:.2f}")
        return results

    signals = generate_weather_signals(
        brackets=brackets,
        bracket_probs=bracket_probs,
        city="National ILI",
        bankroll=bankroll,
        min_edge=0.08,
        max_position_pct=0.05,
        kelly_fraction=0.5,
        rationale_prefix=f"FLU({consensus.confidence})",
    )

    results["signals_generated"] = len(signals)

    if not signals:
        logger.info("FLU CONSENSUS: No trade signals generated")
        return results

    signals.sort(key=lambda s: s.edge, reverse=True)
    max_trades = 3

    logger.info(f"FLU CONSENSUS: Top signals ({len(signals)} total):")
    for i, sig in enumerate(signals[:max_trades]):
        logger.info(f"  #{i+1}: {sig.rationale}")

    # 6. Execute trades
    for signal in signals[:max_trades]:
        if paper_mode:
            log_signal(
                market_id=signal.bracket.ticker,
                market_title=signal.rationale,
                side=signal.side,
                entry_price=signal.limit_price / 100.0,
                confidence=signal.confidence,
                reasoning=signal.rationale,
                strategy=strategy_tag,
            )
            results["orders_placed"] += 1
            logger.info(f"FLU PAPER TRADE: {signal.bracket.ticker} {signal.side} @ {signal.limit_price}c")
        else:
            success = await execute_weather_trade(
                signal, kalshi_client, db_manager,
                strategy=strategy_tag,
            )
            if success:
                results["orders_placed"] += 1
                results["total_position_value"] += signal.position_size_dollars

    logger.info(
        f"FLU CONSENSUS cycle complete: {results['brackets_found']} brackets, "
        f"{results['signals_generated']} signals, {results['orders_placed']} orders "
        f"({'PAPER' if paper_mode else 'LIVE'})"
    )
    return results
