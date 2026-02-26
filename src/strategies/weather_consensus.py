"""
Weather Consensus Trading Strategy

Uses multi-source forecast consensus (NWS + Open-Meteo GFS + ECMWF) to
trade Kalshi temperature markets. Only trades when sources agree; tighter
sigma (higher confidence) when consensus is strong.

Falls back to NWS-only single-source logic when consensus is not possible.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.clients.nws_client import WEATHER_STATIONS, WeatherStation
from src.clients.weather_forecast_client import (
    MultiSourceForecast,
    fetch_all_forecasts,
)
from src.clients.kalshi_client import KalshiClient
from src.strategies.weather_strategy import (
    WeatherTradeSignal,
    discover_weather_markets,
    forecast_to_bracket_probs,
    generate_weather_signals,
    execute_weather_trade,
    get_hourly_forecast,
    get_forecast_high_for_date,
)
from src.utils.database import DatabaseManager
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("weather_consensus")


# ============================================================
# Consensus data structures
# ============================================================

@dataclass
class ConsensusResult:
    """Result of the multi-source consensus algorithm."""
    consensus_temp: float          # Median of agreeing cluster
    sigma: float                   # Gaussian sigma to use
    confidence: str                # "high", "medium", "low", "skip"
    agreement_ratio: float         # fraction of sources that agree
    cluster_temps: List[float]     # temperatures in the agreeing cluster
    all_temps: List[float]         # all reported temperatures
    source_count: int              # total sources available


# ============================================================
# Consensus algorithm
# ============================================================

_CLUSTER_TOLERANCE = 2.0  # °F — sources within this range are "agreeing"


def compute_consensus(forecast: MultiSourceForecast) -> ConsensusResult:
    """
    Determine consensus temperature and confidence from multi-source forecasts.

    Algorithm:
      1. Collect all forecast highs.
      2. If <3 sources → confidence="skip" (fall back to NWS-only).
      3. Sort temps, find largest cluster within ±2°F tolerance.
      4. Map agreement_ratio to sigma:
         - >=80% agree → sigma=1.5 ("high")
         - >=60% agree → sigma=2.5 ("medium")
         - below        → sigma=4.0 ("low"), skip trading
      5. Time-of-day adjustment: +0.5 before 10 AM, +1.0 before 6 AM.
      6. Consensus temp = median of agreeing cluster.
    """
    temps = [s.temperature_high_f for s in forecast.sources]
    n = len(temps)

    if n < 3:
        # Not enough sources for consensus
        fallback_temp = temps[0] if temps else 0.0
        return ConsensusResult(
            consensus_temp=fallback_temp,
            sigma=3.0,
            confidence="skip",
            agreement_ratio=0.0,
            cluster_temps=temps,
            all_temps=temps,
            source_count=n,
        )

    sorted_temps = sorted(temps)

    # Sliding window: find the largest cluster within tolerance
    best_cluster: List[float] = []
    for i in range(n):
        cluster = [sorted_temps[i]]
        for j in range(i + 1, n):
            if sorted_temps[j] - sorted_temps[i] <= _CLUSTER_TOLERANCE:
                cluster.append(sorted_temps[j])
            else:
                break
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    agreement_ratio = len(best_cluster) / n

    # Map ratio to sigma and confidence label
    if agreement_ratio >= 0.80:
        sigma = 1.5
        confidence = "high"
    elif agreement_ratio >= 0.60:
        sigma = 2.5
        confidence = "medium"
    else:
        sigma = 4.0
        confidence = "low"

    # Time-of-day adjustment: forecasts are less certain early in the morning
    hour = datetime.now().hour
    if hour < 6:
        sigma += 1.0
    elif hour < 10:
        sigma += 0.5

    # Consensus temp = median of the agreeing cluster
    best_cluster.sort()
    mid = len(best_cluster) // 2
    if len(best_cluster) % 2 == 0:
        consensus_temp = (best_cluster[mid - 1] + best_cluster[mid]) / 2.0
    else:
        consensus_temp = best_cluster[mid]

    return ConsensusResult(
        consensus_temp=consensus_temp,
        sigma=sigma,
        confidence=confidence,
        agreement_ratio=agreement_ratio,
        cluster_temps=best_cluster,
        all_temps=temps,
        source_count=n,
    )


# ============================================================
# Single-source fallback (original NWS-only logic)
# ============================================================

async def _nws_fallback_for_city(
    station: WeatherStation,
    kalshi_client: KalshiClient,
    db_manager: DatabaseManager,
    weather_bankroll: float,
    target_date: str,
) -> List[WeatherTradeSignal]:
    """Run original single-source weather logic for one city."""
    periods = await get_hourly_forecast(station)
    if not periods:
        return []
    forecast_high = get_forecast_high_for_date(periods, target_date)
    if forecast_high is None:
        return []

    brackets = await discover_weather_markets(kalshi_client, station, target_date)
    if not brackets:
        return []

    hour = datetime.now().hour
    if hour >= 10:
        sigma = 2.0
    elif hour >= 6:
        sigma = 2.5
    else:
        sigma = 3.5

    bracket_probs = forecast_to_bracket_probs(forecast_high, brackets, sigma=sigma)
    return generate_weather_signals(
        brackets=brackets,
        bracket_probs=bracket_probs,
        city=station.city,
        bankroll=weather_bankroll,
        min_edge=0.08,
        max_position_pct=0.05,
        kelly_fraction=0.5,
        rationale_prefix="WEATHER-NWS",
    )


# ============================================================
# Main consensus trading cycle
# ============================================================

async def run_consensus_weather_cycle(
    kalshi_client: KalshiClient,
    db_manager: DatabaseManager,
) -> Dict:
    """
    Run one complete weather-consensus trading cycle.

    For each city:
      1. fetch_all_forecasts (3 sources)
      2. compute_consensus
      3. discover Kalshi weather markets
      4. forecast_to_bracket_probs with consensus temp + sigma
      5. generate_weather_signals
      6. execute trades

    Falls back to NWS-only when consensus is impossible.

    Returns summary dict matching the interface of run_weather_trading_cycle.
    """
    logger.info("CONSENSUS: Starting weather consensus trading cycle...")

    results: Dict = {
        "cities_analyzed": 0,
        "brackets_found": 0,
        "signals_generated": 0,
        "orders_placed": 0,
        "total_position_value": 0.0,
    }

    # Get available balance
    try:
        balance_response = await kalshi_client.get_balance()
        bankroll = balance_response.get("balance", 0) / 100.0
    except Exception as e:
        logger.error(f"Could not fetch balance: {e}")
        return results

    # Use full available balance — no new deposits, trade only with what's in the account
    weather_bankroll = bankroll
    if weather_bankroll < 5.0:
        logger.warning(f"Insufficient weather bankroll: ${weather_bankroll:.2f}")
        return results

    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    target_date = today

    all_signals: List[WeatherTradeSignal] = []

    for city_key, station in WEATHER_STATIONS.items():
        try:
            results["cities_analyzed"] += 1

            # 1. Multi-source forecast
            forecast = await fetch_all_forecasts(station, target_date)

            # 2. Consensus
            consensus = compute_consensus(forecast)

            logger.info(
                f"CONSENSUS {station.city}: confidence={consensus.confidence}, "
                f"temp={consensus.consensus_temp:.0f}F, sigma={consensus.sigma:.1f}, "
                f"agree={consensus.agreement_ratio:.0%} "
                f"(temps={[f'{t:.0f}' for t in consensus.all_temps]})"
            )

            # 3. Low confidence or skip → try fallback
            if consensus.confidence in ("low", "skip"):
                if consensus.confidence == "skip" and consensus.source_count > 0:
                    logger.info(f"CONSENSUS {station.city}: falling back to NWS-only")
                    fallback = await _nws_fallback_for_city(
                        station, kalshi_client, db_manager, weather_bankroll, target_date,
                    )
                    all_signals.extend(fallback)
                else:
                    logger.info(f"CONSENSUS {station.city}: skipping (low confidence)")
                await asyncio.sleep(1)
                continue

            # 4. Discover Kalshi markets
            brackets = await discover_weather_markets(kalshi_client, station, target_date)
            results["brackets_found"] += len(brackets)
            if not brackets:
                logger.info(f"CONSENSUS {station.city}: no active brackets")
                await asyncio.sleep(1)
                continue

            # 5. Bracket probabilities using consensus temp + sigma
            bracket_probs = forecast_to_bracket_probs(
                consensus.consensus_temp, brackets, sigma=consensus.sigma,
            )

            # 6. Generate signals
            city_signals = generate_weather_signals(
                brackets=brackets,
                bracket_probs=bracket_probs,
                city=station.city,
                bankroll=weather_bankroll,
                min_edge=0.08,
                max_position_pct=0.05,
                kelly_fraction=0.5,
                rationale_prefix=f"CONSENSUS({consensus.confidence})",
            )

            if city_signals:
                logger.info(
                    f"CONSENSUS {station.city}: {len(city_signals)} signals "
                    f"(best edge: {city_signals[0].edge:.0%})"
                )
                all_signals.extend(city_signals)

            await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error in consensus analysis for {station.city}: {e}")
            continue

    results["signals_generated"] = len(all_signals)

    if not all_signals:
        logger.info("CONSENSUS: No trade signals generated this cycle")
        return results

    # Sort by edge, take top opportunities
    all_signals.sort(key=lambda s: s.edge, reverse=True)
    max_trades = 5

    # Filter out signals for tickers where we already hold a position
    held_tickers = set()
    try:
        open_positions = await db_manager.get_open_live_positions()
        held_tickers = {p.market_id for p in open_positions}
        if held_tickers:
            before = len(all_signals)
            all_signals = [s for s in all_signals if s.bracket.ticker not in held_tickers]
            skipped = before - len(all_signals)
            if skipped > 0:
                logger.info(f"CONSENSUS: Filtered out {skipped} signals for already-held positions")
    except Exception as e:
        logger.warning(f"Could not check existing positions: {e}")

    logger.info(f"CONSENSUS: Top signals ({len(all_signals)} total):")
    for i, sig in enumerate(all_signals[:max_trades]):
        logger.info(f"  #{i+1}: {sig.rationale}")

    # Execute
    for signal in all_signals[:max_trades]:
        success = await execute_weather_trade(
            signal, kalshi_client, db_manager,
            strategy="weather_consensus",
        )
        if success:
            results["orders_placed"] += 1
            results["total_position_value"] += signal.position_size_dollars

    logger.info(
        f"CONSENSUS cycle complete: {results['cities_analyzed']} cities, "
        f"{results['brackets_found']} brackets, {results['signals_generated']} signals, "
        f"{results['orders_placed']} orders placed (${results['total_position_value']:.2f})"
    )

    return results
