"""
Economic Data Consensus Trading Strategy

Uses multi-source economic nowcast consensus (Atlanta Fed + Cleveland Fed +
NY Fed) to trade Kalshi economic indicator bracket markets (CPI, NFP, GDP).

Mirrors the weather consensus pattern with econ-specific parameters.
"""

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from src.clients.econ_forecast_client import (
    ECON_INDICATORS,
    EconIndicator,
    MultiSourceForecast,
    fetch_econ_forecasts,
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

logger = get_trading_logger("econ_consensus")


# ============================================================
# Consensus algorithm
# ============================================================

@dataclass
class EconConsensusResult:
    """Result of economic data multi-source consensus."""
    consensus_value: float
    sigma: float
    confidence: str        # "high", "medium", "low", "skip"
    agreement_ratio: float
    cluster_values: List[float]
    all_values: List[float]
    source_count: int
    indicator: str


def compute_econ_consensus(
    forecast: MultiSourceForecast,
    indicator: EconIndicator,
) -> EconConsensusResult:
    """
    Determine consensus economic value and confidence from multi-source data.

    Algorithm mirrors weather consensus with indicator-specific tolerances.
    """
    values = [s.value for s in forecast.sources]
    n = len(values)

    if n == 0:
        return EconConsensusResult(
            consensus_value=0.0,
            sigma=indicator.sigma_low,
            confidence="skip",
            agreement_ratio=0.0,
            cluster_values=[],
            all_values=[],
            source_count=0,
            indicator=indicator.name,
        )

    if n == 1:
        # Single source: treat as medium confidence with wider sigma
        return EconConsensusResult(
            consensus_value=values[0],
            sigma=indicator.sigma_med,
            confidence="medium",
            agreement_ratio=1.0,
            cluster_values=values,
            all_values=values,
            source_count=1,
            indicator=indicator.name,
        )

    sorted_values = sorted(values)
    tolerance = indicator.cluster_tolerance

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
        sigma = indicator.sigma_high
        confidence = "high"
    elif agreement_ratio >= 0.60:
        sigma = indicator.sigma_med
        confidence = "medium"
    else:
        sigma = indicator.sigma_low
        confidence = "low"

    # Consensus = median of agreeing cluster
    best_cluster.sort()
    mid = len(best_cluster) // 2
    if len(best_cluster) % 2 == 0:
        consensus_value = (best_cluster[mid - 1] + best_cluster[mid]) / 2.0
    else:
        consensus_value = best_cluster[mid]

    return EconConsensusResult(
        consensus_value=consensus_value,
        sigma=sigma,
        confidence=confidence,
        agreement_ratio=agreement_ratio,
        cluster_values=best_cluster,
        all_values=values,
        source_count=n,
        indicator=indicator.name,
    )


# ============================================================
# Market discovery + bracket parsing
# ============================================================

async def _discover_econ_markets(
    kalshi_client: KalshiClient,
    indicator: EconIndicator,
) -> List[TemperatureBracket]:
    """
    Discover active economic bracket markets on Kalshi.
    Tries multiple series ticker candidates for the given indicator.
    """
    brackets = []

    for series in indicator.kalshi_series_candidates:
        try:
            markets_response = await kalshi_client.get_markets(
                limit=100,
                series_ticker=series,
            )
            markets = markets_response.get("markets", [])
            if not markets:
                continue

            logger.info(f"Found {len(markets)} markets for series '{series}' ({indicator.name})")

            for market in markets:
                if market.get("status") != "active":
                    continue
                bracket = _parse_econ_bracket(market, indicator)
                if bracket:
                    brackets.append(bracket)

            if brackets:
                break

        except Exception as e:
            logger.debug(f"Series '{series}' lookup failed: {e}")
            continue

    logger.info(f"Econ markets ({indicator.name}): {len(brackets)} active brackets discovered")
    return brackets


def _parse_econ_bracket(market: dict, indicator: EconIndicator) -> Optional[TemperatureBracket]:
    """
    Parse an economic bracket from a Kalshi market.

    Expected title formats:
      CPI: ">X%", "<X%", "X%-Y%", "X.X% to Y.Y%"
      NFP: ">XK", "<XK", "XK-YK", "X,000 to Y,000"
      GDP: ">X%", "<X%", "X%-Y%"
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

    if indicator.name == "NFP":
        # NFP brackets: "XK-YK", ">XK", "<XK", "X,000 to Y,000"
        range_k = re.search(r'(\d+)\s*[Kk]\s*[-–to]+\s*(\d+)\s*[Kk]', title)
        range_full = re.search(r'(\d[\d,]*)\s*[-–to]+\s*(\d[\d,]*)', title)
        above_k = re.search(r'(?:above|>|over|at least)\s*(\d+)\s*[Kk]', title, re.IGNORECASE)
        below_k = re.search(r'(?:below|<|under|at most)\s*(\d+)\s*[Kk]', title, re.IGNORECASE)
        above_full = re.search(r'(?:above|>|over|at least)\s*(\d[\d,]*)', title, re.IGNORECASE)
        below_full = re.search(r'(?:below|<|under|at most)\s*(\d[\d,]*)', title, re.IGNORECASE)

        if range_k:
            low = int(range_k.group(1))
            high = int(range_k.group(2))
        elif above_k and not below_k:
            low = int(above_k.group(1))
        elif below_k and not above_k:
            high = int(below_k.group(1))
        elif range_full:
            low = int(range_full.group(1).replace(",", "")) // 1000
            high = int(range_full.group(2).replace(",", "")) // 1000
        elif above_full and not below_full:
            low = int(above_full.group(1).replace(",", "")) // 1000
        elif below_full and not above_full:
            high = int(below_full.group(1).replace(",", "")) // 1000
    else:
        # CPI/GDP: percentage brackets
        # Scale: multiply by 100 to get integer basis points for bracket math
        # e.g. 2.5% → 250 bps
        range_pct = re.search(r'(-?\d+\.?\d*)\s*%?\s*[-–to]+\s*(-?\d+\.?\d*)\s*%', title)
        above_pct = re.search(r'(?:above|>|over|at least)\s*(-?\d+\.?\d*)\s*%', title, re.IGNORECASE)
        below_pct = re.search(r'(?:below|<|under|at most)\s*(-?\d+\.?\d*)\s*%', title, re.IGNORECASE)

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
                f"ECON auto-switch: {len(rows)} settled, "
                f"win_rate={win_rate:.1f}%, pnl=${total_pnl:.2f} — switching to LIVE"
            )
            return True
        return False
    except Exception:
        return False


# ============================================================
# Main econ consensus trading cycle
# ============================================================

async def run_econ_consensus_cycle(
    kalshi_client: KalshiClient,
    db_manager: DatabaseManager,
    paper_mode: bool = True,
) -> Dict:
    """
    Run one complete economic data consensus trading cycle.

    Iterates over all configured indicators (CPI, NFP, GDP):
    1. Fetch economic nowcasts from 3 sources
    2. Compute consensus
    3. Discover Kalshi economic markets
    4. Convert consensus to bracket probabilities
    5. Generate trade signals
    6. Execute trades (paper or live)
    """
    strategy_tag = "econ_consensus"
    logger.info(f"ECON CONSENSUS: Starting cycle (paper={paper_mode})...")

    # Auto-switch check
    if paper_mode and _check_paper_performance(strategy_tag):
        paper_mode = False
        logger.info("ECON CONSENSUS: Auto-switched to LIVE mode!")

    results: Dict = {
        "indicators_analyzed": 0,
        "brackets_found": 0,
        "signals_generated": 0,
        "orders_placed": 0,
        "total_position_value": 0.0,
        "paper_mode": paper_mode,
    }

    # Get available balance
    try:
        balance_response = await kalshi_client.get_balance()
        bankroll = balance_response.get("balance", 0) / 100.0
    except Exception as e:
        logger.error(f"Could not fetch balance: {e}")
        return results

    if bankroll < 5.0:
        logger.warning(f"ECON CONSENSUS: Insufficient bankroll: ${bankroll:.2f}")
        return results

    all_signals: List[WeatherTradeSignal] = []

    for indicator_name, indicator in ECON_INDICATORS.items():
        try:
            results["indicators_analyzed"] += 1

            # 1. Fetch economic forecasts
            forecast = await fetch_econ_forecasts(indicator_name)
            if not forecast.sources:
                logger.warning(f"ECON CONSENSUS ({indicator_name}): No sources available")
                continue

            # 2. Compute consensus
            consensus = compute_econ_consensus(forecast, indicator)

            logger.info(
                f"ECON CONSENSUS ({indicator_name}): confidence={consensus.confidence}, "
                f"value={consensus.consensus_value:.2f}{indicator.unit}, "
                f"sigma={consensus.sigma:.3f}, agree={consensus.agreement_ratio:.0%} "
                f"(values={[f'{v:.2f}' for v in consensus.all_values]})"
            )

            if consensus.confidence in ("low", "skip"):
                logger.info(f"ECON CONSENSUS ({indicator_name}): skipping (low confidence)")
                await asyncio.sleep(1)
                continue

            # 3. Discover Kalshi markets
            brackets = await _discover_econ_markets(kalshi_client, indicator)
            results["brackets_found"] += len(brackets)

            if not brackets:
                logger.info(f"ECON CONSENSUS ({indicator_name}): no active brackets found")
                await asyncio.sleep(1)
                continue

            # 4. Calculate bracket probabilities
            # For CPI/GDP: consensus_value is in %, brackets are in basis points (x100)
            # For NFP: consensus_value is in K, brackets are in K
            if indicator_name in ("CPI", "GDP"):
                consensus_for_brackets = int(round(consensus.consensus_value * 100))
                sigma_for_brackets = consensus.sigma * 100
            else:
                consensus_for_brackets = consensus.consensus_value
                sigma_for_brackets = consensus.sigma

            bracket_probs = forecast_to_bracket_probs(
                consensus_for_brackets, brackets, sigma=sigma_for_brackets,
            )

            # 5. Generate trade signals
            indicator_signals = generate_weather_signals(
                brackets=brackets,
                bracket_probs=bracket_probs,
                city=indicator_name,
                bankroll=bankroll,
                min_edge=0.08,
                max_position_pct=0.05,
                kelly_fraction=0.5,
                rationale_prefix=f"ECON-{indicator_name}({consensus.confidence})",
            )

            if indicator_signals:
                logger.info(
                    f"ECON CONSENSUS ({indicator_name}): {len(indicator_signals)} signals "
                    f"(best edge: {indicator_signals[0].edge:.0%})"
                )
                all_signals.extend(indicator_signals)

            await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error in econ consensus for {indicator_name}: {e}")
            continue

    results["signals_generated"] = len(all_signals)

    if not all_signals:
        logger.info("ECON CONSENSUS: No trade signals generated this cycle")
        return results

    # Sort by edge, take top opportunities
    all_signals.sort(key=lambda s: s.edge, reverse=True)
    max_trades = 5

    logger.info(f"ECON CONSENSUS: Top signals ({len(all_signals)} total):")
    for i, sig in enumerate(all_signals[:max_trades]):
        logger.info(f"  #{i+1}: {sig.rationale}")

    # 6. Execute trades
    for signal in all_signals[:max_trades]:
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
            logger.info(f"ECON PAPER TRADE: {signal.bracket.ticker} {signal.side} @ {signal.limit_price}c")
        else:
            success = await execute_weather_trade(
                signal, kalshi_client, db_manager,
                strategy=strategy_tag,
            )
            if success:
                results["orders_placed"] += 1
                results["total_position_value"] += signal.position_size_dollars

    logger.info(
        f"ECON CONSENSUS cycle complete: {results['indicators_analyzed']} indicators, "
        f"{results['brackets_found']} brackets, {results['signals_generated']} signals, "
        f"{results['orders_placed']} orders ({'PAPER' if paper_mode else 'LIVE'})"
    )
    return results
