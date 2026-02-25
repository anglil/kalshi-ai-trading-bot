"""
Gas Price Consensus Trading Strategy

Uses multi-source gas price consensus (EIA + FRED + AAA) to trade Kalshi
gas price bracket markets. Only trades when sources agree; tighter sigma
(higher confidence) when consensus is strong.

Mirrors the weather consensus pattern with gas-specific parameters.
"""

import asyncio
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from src.clients.gas_price_client import MultiSourceForecast, fetch_gas_forecasts
from src.clients.kalshi_client import KalshiClient
from src.strategies.weather_strategy import (
    TemperatureBracket,
    WeatherTradeSignal,
    forecast_to_bracket_probs,
    generate_weather_signals,
    execute_weather_trade,
    kalshi_taker_fee,
)
from src.paper.tracker import log_signal, get_connection as get_paper_db
from src.utils.database import DatabaseManager
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("gas_consensus")


# ============================================================
# Consensus parameters for gas prices
# ============================================================

GAS_CONSENSUS_PARAMS = {
    "cluster_tolerance": 0.05,   # $0.05 — sources within this range agree
    "sigma_high": 0.03,         # $ — tight sigma when high agreement
    "sigma_med": 0.06,          # $ — medium sigma
    "sigma_low": 0.12,          # $ — wide sigma when low agreement
}

# Series tickers to try on Kalshi for gas markets
GAS_SERIES_CANDIDATES = ["GAS", "KXGAS", "GASPRICE", "GASNATIONAL", "GASOLINE"]


# ============================================================
# Consensus algorithm (reuses weather pattern)
# ============================================================

@dataclass
class GasConsensusResult:
    """Result of gas price multi-source consensus."""
    consensus_price: float       # median of agreeing cluster
    sigma: float                 # Gaussian sigma to use
    confidence: str              # "high", "medium", "low", "skip"
    agreement_ratio: float
    cluster_prices: List[float]
    all_prices: List[float]
    source_count: int


def compute_gas_consensus(forecast: MultiSourceForecast) -> GasConsensusResult:
    """
    Determine consensus gas price and confidence from multi-source data.

    Algorithm mirrors weather consensus:
      1. Collect all gas prices
      2. If <2 sources → confidence="skip"
      3. Sort prices, find largest cluster within tolerance
      4. Map agreement_ratio to sigma
    """
    prices = [s.value for s in forecast.sources]
    n = len(prices)

    if n < 2:
        fallback_price = prices[0] if prices else 0.0
        return GasConsensusResult(
            consensus_price=fallback_price,
            sigma=GAS_CONSENSUS_PARAMS["sigma_low"],
            confidence="skip",
            agreement_ratio=0.0,
            cluster_prices=prices,
            all_prices=prices,
            source_count=n,
        )

    sorted_prices = sorted(prices)
    tolerance = GAS_CONSENSUS_PARAMS["cluster_tolerance"]

    # Sliding window: find largest cluster within tolerance
    best_cluster: List[float] = []
    for i in range(n):
        cluster = [sorted_prices[i]]
        for j in range(i + 1, n):
            if sorted_prices[j] - sorted_prices[i] <= tolerance:
                cluster.append(sorted_prices[j])
            else:
                break
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    agreement_ratio = len(best_cluster) / n

    if agreement_ratio >= 0.80:
        sigma = GAS_CONSENSUS_PARAMS["sigma_high"]
        confidence = "high"
    elif agreement_ratio >= 0.60:
        sigma = GAS_CONSENSUS_PARAMS["sigma_med"]
        confidence = "medium"
    else:
        sigma = GAS_CONSENSUS_PARAMS["sigma_low"]
        confidence = "low"

    # Consensus price = median of agreeing cluster
    best_cluster.sort()
    mid = len(best_cluster) // 2
    if len(best_cluster) % 2 == 0:
        consensus_price = (best_cluster[mid - 1] + best_cluster[mid]) / 2.0
    else:
        consensus_price = best_cluster[mid]

    return GasConsensusResult(
        consensus_price=consensus_price,
        sigma=sigma,
        confidence=confidence,
        agreement_ratio=agreement_ratio,
        cluster_prices=best_cluster,
        all_prices=prices,
        source_count=n,
    )


# ============================================================
# Market discovery + bracket parsing
# ============================================================

async def _discover_gas_markets(kalshi_client: KalshiClient) -> List[TemperatureBracket]:
    """
    Discover active gas price bracket markets on Kalshi.
    Tries multiple series ticker candidates.
    """
    brackets = []

    for series in GAS_SERIES_CANDIDATES:
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
                bracket = _parse_gas_bracket(market)
                if bracket:
                    brackets.append(bracket)

            if brackets:
                break  # Found markets, stop trying other series

        except Exception as e:
            logger.debug(f"Series '{series}' lookup failed: {e}")
            continue

    logger.info(f"Gas markets: {len(brackets)} active brackets discovered")
    return brackets


def _parse_gas_bracket(market: dict) -> Optional[TemperatureBracket]:
    """
    Parse a gas price bracket from a Kalshi market.

    Expected title formats:
      - "above $X.XX"
      - "below $X.XX"
      - "$X.XX-$Y.YY" or "$X.XX to $Y.YY"
      - ">$X.XX" / "<$X.XX"
    """
    title = market.get("title", "")
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

    # Try "above $X.XX" or ">$X.XX" or "> $X.XX"
    above_match = re.search(r'(?:above|>|over|at least)\s*\$?(\d+\.?\d*)', title, re.IGNORECASE)
    below_match = re.search(r'(?:below|<|under|at most)\s*\$?(\d+\.?\d*)', title, re.IGNORECASE)
    range_match = re.search(r'\$(\d+\.?\d*)\s*[-–to]+\s*\$?(\d+\.?\d*)', title, re.IGNORECASE)

    if range_match:
        # Convert dollar values to cents for bracket math
        # We use 100x scale: $3.50 → 350 cents
        low_val = float(range_match.group(1))
        high_val = float(range_match.group(2))
        # Store as integer cents (hundredths of dollar)
        low = int(round(low_val * 100))
        high = int(round(high_val * 100))
    elif above_match and not below_match:
        val = float(above_match.group(1))
        low = int(round(val * 100))
    elif below_match and not above_match:
        val = float(below_match.group(1))
        high = int(round(val * 100))

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
                f"GAS auto-switch: {len(rows)} settled, "
                f"win_rate={win_rate:.1f}%, pnl=${total_pnl:.2f} — switching to LIVE"
            )
            return True
        return False
    except Exception:
        return False


# ============================================================
# Main gas consensus trading cycle
# ============================================================

async def run_gas_consensus_cycle(
    kalshi_client: KalshiClient,
    db_manager: DatabaseManager,
    paper_mode: bool = True,
) -> Dict:
    """
    Run one complete gas price consensus trading cycle.

    1. Fetch gas prices from 3 sources
    2. Compute consensus
    3. Discover Kalshi gas markets
    4. Convert consensus to bracket probabilities
    5. Generate trade signals
    6. Execute trades (paper or live)
    """
    strategy_tag = "gas_consensus"
    logger.info(f"GAS CONSENSUS: Starting cycle (paper={paper_mode})...")

    # Auto-switch check
    if paper_mode and _check_paper_performance(strategy_tag):
        paper_mode = False
        logger.info("GAS CONSENSUS: Auto-switched to LIVE mode!")

    results: Dict = {
        "markets_found": 0,
        "brackets_found": 0,
        "signals_generated": 0,
        "orders_placed": 0,
        "total_position_value": 0.0,
        "paper_mode": paper_mode,
    }

    # 1. Fetch gas prices
    forecast = await fetch_gas_forecasts()
    if not forecast.sources:
        logger.warning("GAS CONSENSUS: No gas price sources available")
        return results

    # 2. Compute consensus
    consensus = compute_gas_consensus(forecast)
    # Convert consensus price to cents for bracket math
    consensus_price_cents = int(round(consensus.consensus_price * 100))

    logger.info(
        f"GAS CONSENSUS: confidence={consensus.confidence}, "
        f"price=${consensus.consensus_price:.3f}, sigma=${consensus.sigma:.3f}, "
        f"agree={consensus.agreement_ratio:.0%} "
        f"(prices={['${:.3f}'.format(p) for p in consensus.all_prices]})"
    )

    if consensus.confidence in ("low", "skip"):
        logger.info("GAS CONSENSUS: skipping (low confidence)")
        return results

    # 3. Discover Kalshi gas markets
    brackets = await _discover_gas_markets(kalshi_client)
    results["brackets_found"] = len(brackets)

    if not brackets:
        logger.info("GAS CONSENSUS: no active gas brackets found")
        return results

    # 4. Calculate bracket probabilities
    # sigma needs to be in the same scale as brackets (cents)
    sigma_cents = consensus.sigma * 100
    bracket_probs = forecast_to_bracket_probs(
        consensus_price_cents, brackets, sigma=sigma_cents,
    )

    # 5. Generate trade signals
    try:
        balance_response = await kalshi_client.get_balance()
        bankroll = balance_response.get("balance", 0) / 100.0
    except Exception as e:
        logger.error(f"Could not fetch balance: {e}")
        return results

    if bankroll < 5.0:
        logger.warning(f"GAS CONSENSUS: Insufficient bankroll: ${bankroll:.2f}")
        return results

    signals = generate_weather_signals(
        brackets=brackets,
        bracket_probs=bracket_probs,
        city="National Gas",
        bankroll=bankroll,
        min_edge=0.08,
        max_position_pct=0.05,
        kelly_fraction=0.5,
        rationale_prefix=f"GAS({consensus.confidence})",
    )

    results["signals_generated"] = len(signals)

    if not signals:
        logger.info("GAS CONSENSUS: No trade signals generated")
        return results

    signals.sort(key=lambda s: s.edge, reverse=True)
    max_trades = 3

    logger.info(f"GAS CONSENSUS: Top signals ({len(signals)} total):")
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
            logger.info(f"GAS PAPER TRADE: {signal.bracket.ticker} {signal.side} @ {signal.limit_price}c")
        else:
            success = await execute_weather_trade(
                signal, kalshi_client, db_manager,
                strategy=strategy_tag,
            )
            if success:
                results["orders_placed"] += 1
                results["total_position_value"] += signal.position_size_dollars

    logger.info(
        f"GAS CONSENSUS cycle complete: {results['brackets_found']} brackets, "
        f"{results['signals_generated']} signals, {results['orders_placed']} orders "
        f"({'PAPER' if paper_mode else 'LIVE'})"
    )
    return results
