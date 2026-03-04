"""
WTI Crude Oil Consensus Trading Strategy

Uses 3-source majority-vote consensus to trade Kalshi WTI crude oil
price bracket markets:
  1. EIA STEO — government short-term energy outlook forecast
  2. Yahoo Finance CL=F — market-implied WTI futures price
  3. FRED DCOILWTICO — WTI spot price with historical volatility

All sources produce point estimates ($/barrel) that are converted to
probability distributions over Kalshi price brackets using a Gaussian model.

Mirrors the econ consensus pattern with oil-specific parameters.
"""

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from src.clients.oil_price_client import (
    OilPriceForecast,
    MultiSourceOilForecast,
    fetch_oil_price_forecasts,
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

logger = get_trading_logger("oil_consensus")


# ============================================================
# Oil-specific parameters
# ============================================================

# Cluster tolerance for consensus: sources within $2/bbl agree
OIL_CLUSTER_TOLERANCE = 2.0

# Sigma values for Gaussian bracket probability model (in $/bbl)
# These control how "spread out" the probability distribution is
OIL_SIGMA_HIGH = 1.0    # High agreement: tight distribution
OIL_SIGMA_MED = 2.0     # Medium agreement: moderate spread
OIL_SIGMA_LOW = 3.5     # Low agreement: wide spread


# ============================================================
# Consensus algorithm
# ============================================================

@dataclass
class OilConsensusResult:
    """Result of WTI oil price multi-source consensus."""
    consensus_price: float   # $/barrel
    sigma: float             # For Gaussian model
    confidence: str          # "high", "medium", "low", "skip"
    agreement_ratio: float
    cluster_prices: List[float]
    all_prices: List[float]
    source_count: int
    avg_volatility: Optional[float]  # Annualized vol if available


def compute_oil_consensus(
    forecast: MultiSourceOilForecast,
) -> OilConsensusResult:
    """
    Determine consensus WTI oil price and confidence from multi-source data.

    Algorithm mirrors weather/econ consensus with oil-specific tolerances.
    """
    prices = [s.price for s in forecast.sources]
    n = len(prices)

    if n == 0:
        return OilConsensusResult(
            consensus_price=0.0,
            sigma=OIL_SIGMA_LOW,
            confidence="skip",
            agreement_ratio=0.0,
            cluster_prices=[],
            all_prices=[],
            source_count=0,
            avg_volatility=None,
        )

    if n == 1:
        vol = forecast.sources[0].volatility
        return OilConsensusResult(
            consensus_price=prices[0],
            sigma=OIL_SIGMA_MED,
            confidence="medium",
            agreement_ratio=1.0,
            cluster_prices=prices,
            all_prices=prices,
            source_count=1,
            avg_volatility=vol,
        )

    sorted_prices = sorted(prices)
    tolerance = OIL_CLUSTER_TOLERANCE

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
        sigma = OIL_SIGMA_HIGH
        confidence = "high"
    elif agreement_ratio >= 0.60:
        sigma = OIL_SIGMA_MED
        confidence = "medium"
    else:
        sigma = OIL_SIGMA_LOW
        confidence = "low"

    # Use historical volatility to refine sigma if available
    vols = [s.volatility for s in forecast.sources if s.volatility is not None]
    avg_vol = None
    if vols:
        avg_vol = sum(vols) / len(vols)
        # Convert annualized vol to daily dollar vol
        # daily_vol = price * annual_vol / sqrt(252)
        avg_price = sum(prices) / len(prices)
        daily_dollar_vol = avg_price * avg_vol / (252 ** 0.5)
        # Use vol-adjusted sigma (blend with confidence-based sigma)
        sigma = (sigma + daily_dollar_vol) / 2.0

    # Consensus = median of agreeing cluster
    best_cluster.sort()
    mid = len(best_cluster) // 2
    if len(best_cluster) % 2 == 0:
        consensus_price = (best_cluster[mid - 1] + best_cluster[mid]) / 2.0
    else:
        consensus_price = best_cluster[mid]

    return OilConsensusResult(
        consensus_price=consensus_price,
        sigma=sigma,
        confidence=confidence,
        agreement_ratio=agreement_ratio,
        cluster_prices=best_cluster,
        all_prices=prices,
        source_count=n,
        avg_volatility=avg_vol,
    )


# ============================================================
# Market discovery + bracket parsing
# ============================================================

# Kalshi series tickers for WTI crude oil price markets
OIL_SERIES = [
    "KXOIL", "OIL", "KXWTI", "WTI", "KXCRUDEOIL", "CRUDEOIL",
    "OILPRICE", "KXOILPRICE", "CRUDE", "KXCRUDE",
]


async def _discover_oil_markets(
    kalshi_client: KalshiClient,
) -> List[TemperatureBracket]:
    """
    Discover active WTI crude oil price bracket markets on Kalshi.
    """
    brackets = []

    for series in OIL_SERIES:
        try:
            markets_response = await kalshi_client.get_markets(
                limit=100,
                series_ticker=series,
                status="open",
            )
            markets = markets_response.get("markets", [])
            if not markets:
                continue

            logger.info(f"Found {len(markets)} markets for series '{series}'")

            for m in markets[:3]:
                logger.debug(
                    f"RAW MARKET: ticker={m.get('ticker')}, title={m.get('title')!r}, "
                    f"status={m.get('status')}, floor_strike={m.get('floor_strike')}, "
                    f"cap_strike={m.get('cap_strike')}, yes_price={m.get('yes_price')}"
                )

            for market in markets:
                if market.get("status") not in ("active", "open"):
                    continue
                bracket = _parse_oil_bracket(market)
                if bracket:
                    brackets.append(bracket)

            if brackets:
                break

        except Exception as e:
            logger.debug(f"Series '{series}' lookup failed: {e}")
            continue

    logger.info(f"Oil price markets: {len(brackets)} active brackets discovered")
    return brackets


def _parse_oil_bracket(market: dict) -> Optional[TemperatureBracket]:
    """
    Parse a WTI oil price bracket from a Kalshi market.

    Expected title formats:
      "$65-$70" / "$65 to $70"
      "Above $70" / "Below $65"
      "WTI above $70/barrel"
      Range brackets with floor_strike/cap_strike API fields
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

    # Try structured API fields first (floor_strike / cap_strike)
    floor_strike = market.get("floor_strike")
    cap_strike = market.get("cap_strike")
    if floor_strike is not None or cap_strike is not None:
        # Oil prices: strikes are in $/barrel, convert to cents for bracket math
        # Actually, keep in dollars since our consensus is in dollars
        # Use integer cents: $65.00 → 6500
        if floor_strike is not None:
            low = int(round(float(floor_strike) * 100))
        if cap_strike is not None:
            high = int(round(float(cap_strike) * 100))

    # Fall through to regex if structured fields didn't work
    if low is None and high is None:
        # Pattern: "$65-$70" or "$65 to $70"
        range_match = re.search(
            r'\$?\s*(\d+\.?\d*)\s*[-–to]+\s*\$?\s*(\d+\.?\d*)',
            title
        )
        above_match = re.search(
            r'(?:above|>|over|at least|more than)\s*\$?\s*(\d+\.?\d*)',
            title, re.IGNORECASE
        )
        below_match = re.search(
            r'(?:below|<|under|at most|less than)\s*\$?\s*(\d+\.?\d*)',
            title, re.IGNORECASE
        )

        if range_match:
            low = int(round(float(range_match.group(1)) * 100))
            high = int(round(float(range_match.group(2)) * 100))
        elif above_match and not below_match:
            low = int(round(float(above_match.group(1)) * 100))
        elif below_match and not above_match:
            high = int(round(float(below_match.group(1)) * 100))

    if low is None and high is None:
        logger.debug(f"PARSE FAIL: title={title!r}, ticker={ticker}")
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
    Returns True if criteria met: >=10 settled, win_rate>=50%, total_pnl>0.
    """
    try:
        conn = get_paper_db()
        rows = conn.execute(
            "SELECT outcome, pnl FROM signals WHERE strategy = ? AND outcome != 'pending'",
            (strategy,),
        ).fetchall()
        conn.close()

        if len(rows) < 10:
            return False

        wins = sum(1 for r in rows if r["outcome"] == "win")
        win_rate = wins / len(rows) * 100
        total_pnl = sum(r["pnl"] for r in rows if r["pnl"] is not None)

        if win_rate >= 50 and total_pnl > 0:
            logger.info(
                f"OIL auto-switch: {len(rows)} settled, "
                f"win_rate={win_rate:.1f}%, pnl=${total_pnl:.2f} — switching to LIVE"
            )
            return True
        return False
    except Exception:
        return False


# ============================================================
# Main oil consensus trading cycle
# ============================================================

async def run_oil_consensus_cycle(
    kalshi_client: KalshiClient,
    db_manager: DatabaseManager,
    paper_mode: bool = False,
) -> Dict:
    """
    Run one complete WTI crude oil consensus trading cycle.

    1. Fetch oil price forecasts from 3 sources
    2. Compute consensus price and confidence
    3. Discover Kalshi oil price bracket markets
    4. Convert consensus to bracket probabilities (Gaussian model)
    5. Generate trade signals where consensus diverges from market prices
    6. Execute trades (paper or live)
    """
    strategy_tag = "oil_consensus"
    logger.info(f"OIL CONSENSUS: Starting cycle (paper={paper_mode})...")

    results: Dict = {
        "brackets_found": 0,
        "signals_generated": 0,
        "orders_placed": 0,
        "total_position_value": 0.0,
        "paper_mode": paper_mode,
    }

    # 1. Fetch oil price forecasts
    forecast = await fetch_oil_price_forecasts()
    if not forecast.sources:
        logger.warning("OIL CONSENSUS: No sources available")
        return results

    # 2. Compute consensus
    consensus = compute_oil_consensus(forecast)

    logger.info(
        f"OIL CONSENSUS: confidence={consensus.confidence}, "
        f"price=${consensus.consensus_price:.2f}/bbl, "
        f"sigma=${consensus.sigma:.2f}, agree={consensus.agreement_ratio:.0%} "
        f"(prices={[f'${p:.2f}' for p in consensus.all_prices]})"
    )

    if consensus.confidence == "skip":
        logger.info("OIL CONSENSUS: skipping (insufficient sources)")
        return results

    # LOOSENED: low confidence now trades with wider sigma instead of skipping
    if consensus.confidence == "low":
        logger.info(
            f"OIL CONSENSUS: low confidence — trading with wider sigma=${consensus.sigma:.2f}"
        )

    # 3. Discover Kalshi oil markets
    brackets = await _discover_oil_markets(kalshi_client)
    results["brackets_found"] = len(brackets)

    if not brackets:
        logger.info("OIL CONSENSUS: no active brackets found")
        return results

    # 4. Calculate bracket probabilities using Gaussian model
    # Consensus price is in dollars, brackets are in cents (x100)
    consensus_cents = int(round(consensus.consensus_price * 100))
    sigma_cents = consensus.sigma * 100

    # Detect if brackets are cumulative
    all_above = all(b.low is not None and b.high is None for b in brackets)
    all_below = all(b.high is not None and b.low is None for b in brackets)
    is_cumulative = all_above or all_below

    bracket_probs = forecast_to_bracket_probs(
        consensus_cents, brackets, sigma=sigma_cents,
        normalize=not is_cumulative,
    )

    if is_cumulative:
        logger.info("OIL: Cumulative brackets detected — normalization SKIPPED")

    # 5. Get available balance
    if paper_mode:
        bankroll = 1000.0
    else:
        try:
            bankroll = await kalshi_client.get_total_portfolio_value()
        except Exception as e:
            logger.error(f"Could not fetch portfolio value: {e}")
            return results

        if bankroll < 5.0:
            logger.warning(f"OIL CONSENSUS: Insufficient bankroll: ${bankroll:.2f}")
            return results

    # 6. Generate trade signals
    all_signals = generate_weather_signals(
        brackets=brackets,
        bracket_probs=bracket_probs,
        city="OIL",
        bankroll=bankroll,
        min_edge=0.08,         # 8% minimum edge
        max_position_pct=0.05, # 5% max per bracket
        kelly_fraction=0.30,
        rationale_prefix=f"OIL({consensus.confidence})",
        max_shares=5,
    )

    results["signals_generated"] = len(all_signals)

    if not all_signals:
        logger.info("OIL CONSENSUS: No trade signals generated this cycle")
        return results

    # Sort by edge, take top opportunities
    all_signals.sort(key=lambda s: s.edge, reverse=True)
    max_trades = 3  # Conservative for new strategy

    logger.info(f"OIL CONSENSUS: Top signals ({len(all_signals)} total):")
    for i, sig in enumerate(all_signals[:max_trades]):
        logger.info(f"  #{i+1}: {sig.rationale}")

    # 7. Execute trades
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
            logger.info(
                f"OIL PAPER TRADE: {signal.bracket.ticker} {signal.side} "
                f"@ {signal.limit_price}c"
            )
        else:
            success = await execute_weather_trade(
                signal, kalshi_client, db_manager,
                strategy=strategy_tag,
            )
            if success:
                results["orders_placed"] += 1
                results["total_position_value"] += signal.position_size_dollars

    logger.info(
        f"OIL CONSENSUS cycle complete: {results['brackets_found']} brackets, "
        f"{results['signals_generated']} signals, {results['orders_placed']} orders "
        f"({'PAPER' if paper_mode else 'LIVE'})"
    )
    return results
