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
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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
# Risk management constants
# ============================================================

MAX_POSITIONS_PER_CITY = 2       # Max concurrent positions per city (reduced from 3)
BRACKET_OVERLAP_THRESHOLD = 4    # °F — brackets within this range are "overlapping"
MAX_DAILY_LOSSES_PER_CITY = 2    # Stop trading a city after this many consecutive losses in a day
MIN_EDGE_THRESHOLD = 0.15        # 15% minimum edge (raised from 8%)
KELLY_FRACTION = 0.25            # Quarter-Kelly (reduced from 0.5)
MAX_POSITION_PCT = 0.03          # 3% per bracket (reduced from 5%)
MAX_SHARES_PER_TRADE = 5         # Cap shares per trade (reduced from 10)


# ============================================================
# Helpers for ticker parsing
# ============================================================

def _extract_city_from_ticker(ticker: str) -> str:
    """Extract city code from a weather ticker.

    Ticker format: KXHIGH{CITY}-{DATE}-{BRACKET}
    e.g. KXHIGHAUS-26FEB27-B85.5 → AUS
    """
    prefix = ticker.split('-')[0]
    return prefix.replace('KXHIGH', '')


def _parse_bracket_from_ticker(ticker: str) -> Optional[Tuple[str, float]]:
    """Parse bracket type and threshold from the last segment of a ticker.

    e.g. KXHIGHAUS-26FEB27-B85.5 → ("B", 85.5)
         KXHIGHAUS-26FEB27-T83   → ("T", 83.0)

    Returns None if the bracket segment cannot be parsed.
    """
    parts = ticker.split('-')
    if len(parts) < 3:
        return None
    bracket_seg = parts[-1]  # e.g. "B85.5" or "T83"
    if not bracket_seg:
        return None
    bracket_type = bracket_seg[0].upper()  # "B" or "T"
    try:
        threshold = float(bracket_seg[1:])
    except (ValueError, IndexError):
        return None
    return (bracket_type, threshold)


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
    # Priority 5: Calibrate sigma upward — our forecast is less precise than we thought
    if agreement_ratio >= 0.80:
        sigma = 2.5   # Was 1.5 — too tight, caused overconfident bets
        confidence = "high"
    elif agreement_ratio >= 0.60:
        sigma = 3.5   # Was 2.5
        confidence = "medium"
    else:
        sigma = 5.0   # Was 4.0
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
        sigma = 3.0   # Was 2.0 — calibrated upward
    elif hour >= 6:
        sigma = 3.5   # Was 2.5
    else:
        sigma = 4.5   # Was 3.5

    bracket_probs = forecast_to_bracket_probs(forecast_high, brackets, sigma=sigma)
    return generate_weather_signals(
        brackets=brackets,
        bracket_probs=bracket_probs,
        city=station.city,
        bankroll=weather_bankroll,
        min_edge=MIN_EDGE_THRESHOLD,
        max_position_pct=MAX_POSITION_PCT,
        kelly_fraction=KELLY_FRACTION,
        rationale_prefix="WEATHER-NWS",
        max_shares=MAX_SHARES_PER_TRADE,
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

    # Get total portfolio value (cash + positions) for proper bankroll sizing
    try:
        bankroll = await kalshi_client.get_total_portfolio_value()
    except Exception as e:
        logger.error(f"Could not fetch portfolio value: {e}")
        return results

    # Priority 3: Cap weather exposure at 30% of total portfolio
    WEATHER_EXPOSURE_CAP = 0.30  # Max 30% of portfolio for weather
    DAILY_DEPLOYMENT_LIMIT = 50.0  # Max $50/day in new weather buys
    
    # Check ACTUAL weather exposure from Kalshi API (not just bankroll calculation)
    try:
        api_positions = await kalshi_client._make_authenticated_request(
            'GET', '/trade-api/v2/portfolio/positions'
        )
        weather_exposure_cents = 0
        total_exposure_cents = 0
        for mp in api_positions.get('market_positions', []):
            exp = mp.get('market_exposure', 0)
            total_exposure_cents += exp
            ticker = mp.get('ticker', '')
            if 'HIGH' in ticker or 'LOW' in ticker or 'TEMP' in ticker:
                weather_exposure_cents += exp
        
        weather_exposure = weather_exposure_cents / 100
        total_exposure = total_exposure_cents / 100
        weather_pct = weather_exposure / bankroll * 100 if bankroll > 0 else 100
        
        logger.info(
            f"ACTUAL WEATHER EXPOSURE: ${weather_exposure:.2f} ({weather_pct:.0f}% of ${bankroll:.2f} portfolio) "
            f"| Total exposure: ${total_exposure:.2f}"
        )
        
        # Hard block if weather already exceeds cap
        if weather_pct >= WEATHER_EXPOSURE_CAP * 100:
            logger.warning(
                f"WEATHER CAP EXCEEDED: ${weather_exposure:.2f} is {weather_pct:.0f}% of portfolio "
                f"(cap is {WEATHER_EXPOSURE_CAP*100:.0f}%). No new weather trades."
            )
            return results
    except Exception as e:
        logger.warning(f"Could not check API exposure: {e}")
    
    weather_bankroll = bankroll * WEATHER_EXPOSURE_CAP
    if weather_bankroll < 5.0:
        logger.warning(f"Insufficient weather bankroll: ${weather_bankroll:.2f} (30% of ${bankroll:.2f})")
        return results
    logger.info(f"WEATHER BANKROLL: ${weather_bankroll:.2f} (30% cap of ${bankroll:.2f} portfolio)")

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

            # 6. Generate signals (using tightened risk parameters)
            city_signals = generate_weather_signals(
                brackets=brackets,
                bracket_probs=bracket_probs,
                city=station.city,
                bankroll=weather_bankroll,
                min_edge=MIN_EDGE_THRESHOLD,
                max_position_pct=MAX_POSITION_PCT,
                kelly_fraction=KELLY_FRACTION,
                rationale_prefix=f"CONSENSUS({consensus.confidence})",
                max_shares=MAX_SHARES_PER_TRADE,
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
    max_trades = 3  # Reduced from 5 to limit daily deployment

    # Filter out signals for tickers where we already hold a position
    # Use KALSHI API as source of truth (not just local DB which may be incomplete)
    held_tickers = set()
    city_position_counts: Dict[str, int] = defaultdict(int)
    held_brackets: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
    try:
        # Get positions from Kalshi API (source of truth)
        from src.jobs.execute import _check_existing_kalshi_position
        api_positions = await kalshi_client._make_authenticated_request(
            'GET', '/trade-api/v2/portfolio/positions'
        )
        api_market_positions = api_positions.get('market_positions', [])
        
        # Build held_tickers from API positions (any non-zero position)
        for mp in api_market_positions:
            ticker = mp.get('ticker', '')
            pos = mp.get('position', 0)
            if pos != 0 and ('HIGH' in ticker or 'LOW' in ticker or 'TEMP' in ticker):
                held_tickers.add(ticker)
                city = _extract_city_from_ticker(ticker)
                city_position_counts[city] += 1
                parsed = _parse_bracket_from_ticker(ticker)
                if parsed:
                    bracket_type, threshold = parsed
                    side = 'YES' if pos > 0 else 'NO'
                    held_brackets[city].append((side, bracket_type, threshold))
        
        # Also add non-weather held tickers from API
        for mp in api_market_positions:
            ticker = mp.get('ticker', '')
            pos = mp.get('position', 0)
            if pos != 0:
                held_tickers.add(ticker)
        
        # Fallback: also check local DB for any positions the API might miss
        open_positions = await db_manager.get_open_live_positions()
        for p in open_positions:
            held_tickers.add(p.market_id)

        # Build per-city counts and held bracket info for weather positions
        for p in open_positions:
            strat = getattr(p, 'strategy', '') or ''
            if strat.startswith('weather') or strat == 'weather_consensus':
                city = _extract_city_from_ticker(p.market_id)
                city_position_counts[city] += 1

                parsed = _parse_bracket_from_ticker(p.market_id)
                if parsed:
                    bracket_type, threshold = parsed
                    held_brackets[city].append((p.side, bracket_type, threshold))

        if held_tickers:
            before = len(all_signals)
            all_signals = [s for s in all_signals if s.bracket.ticker not in held_tickers]
            skipped = before - len(all_signals)
            if skipped > 0:
                logger.info(f"CONSENSUS: Filtered out {skipped} signals for already-held positions")

        # Fix 1: Per-city concentration limit
        before = len(all_signals)
        filtered_signals = []
        for sig in all_signals:
            city = _extract_city_from_ticker(sig.bracket.ticker)
            if city_position_counts[city] >= MAX_POSITIONS_PER_CITY:
                logger.info(
                    f"CITY LIMIT: Skipping {sig.bracket.ticker} {sig.side} — "
                    f"{city} already has {city_position_counts[city]} positions"
                )
                continue
            filtered_signals.append(sig)
        all_signals = filtered_signals
        city_skipped = before - len(all_signals)
        if city_skipped > 0:
            logger.info(f"CONSENSUS: Filtered out {city_skipped} signals due to per-city limit")

        # Fix 2: Overlap filter — skip brackets that overlap existing held brackets
        before = len(all_signals)
        filtered_signals = []
        for sig in all_signals:
            city = _extract_city_from_ticker(sig.bracket.ticker)
            parsed = _parse_bracket_from_ticker(sig.bracket.ticker)
            if parsed and city in held_brackets:
                sig_type, sig_threshold = parsed
                dominated = False
                for held_side, held_type, held_threshold in held_brackets[city]:
                    if (held_side == sig.side
                            and held_type == sig_type
                            and abs(sig_threshold - held_threshold) <= BRACKET_OVERLAP_THRESHOLD):
                        logger.info(
                            f"OVERLAP SKIP: {sig.bracket.ticker} {sig.side} overlaps "
                            f"existing {held_type}{held_threshold} {held_side} in {city}"
                        )
                        dominated = True
                        break
                if dominated:
                    continue
            filtered_signals.append(sig)
        all_signals = filtered_signals
        overlap_skipped = before - len(all_signals)
        if overlap_skipped > 0:
            logger.info(f"CONSENSUS: Filtered out {overlap_skipped} signals due to bracket overlap")

    except Exception as e:
        logger.warning(f"Could not check existing positions: {e}")

    logger.info(f"CONSENSUS: Top signals ({len(all_signals)} total):")
    for i, sig in enumerate(all_signals[:max_trades]):
        logger.info(f"  #{i+1}: {sig.rationale}")

    # Execute — update city_position_counts within cycle to enforce limits
    for signal in all_signals[:max_trades]:
        city = _extract_city_from_ticker(signal.bracket.ticker)
        if city_position_counts[city] >= MAX_POSITIONS_PER_CITY:
            logger.info(
                f"CITY LIMIT: Skipping {signal.bracket.ticker} {signal.side} — "
                f"{city} hit limit during execution"
            )
            continue
        success = await execute_weather_trade(
            signal, kalshi_client, db_manager,
            strategy="weather_consensus",
        )
        if success:
            results["orders_placed"] += 1
            results["total_position_value"] += signal.position_size_dollars
            city_position_counts[city] += 1

    logger.info(
        f"CONSENSUS cycle complete: {results['cities_analyzed']} cities, "
        f"{results['brackets_found']} brackets, {results['signals_generated']} signals, "
        f"{results['orders_placed']} orders placed (${results['total_position_value']:.2f})"
    )

    return results
