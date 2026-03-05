"""
FOMC Rate Decision Consensus Trading Strategy

Uses 3-source majority-vote consensus to trade Kalshi FOMC rate decision
bracket markets:
  1. FRED Fed Funds Rate + Futures-Implied Probabilities
  2. Atlanta Fed Market Probability Tracker
  3. Investing.com Fed Rate Monitor

All sources produce probability distributions over rate outcomes (hold, cut,
hike) that map directly to Kalshi FOMC bracket markets.

Mirrors the econ consensus pattern with FOMC-specific parameters.
"""

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from src.clients.fed_rate_client import (
    FedRateForecast,
    FedRateOutcome,
    MultiSourceFedForecast,
    fetch_fed_rate_forecasts,
    get_next_fomc_meeting,
    get_days_to_next_meeting,
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

logger = get_trading_logger("fed_rate_consensus")


# ============================================================
# Consensus algorithm
# ============================================================

@dataclass
class FedRateConsensusResult:
    """Result of FOMC rate multi-source consensus."""
    hold_probability: float
    cut_probability: float
    hike_probability: float
    confidence: str          # "high", "medium", "low", "skip"
    agreement_ratio: float
    source_count: int
    meeting_date: str
    current_rate_low: float
    current_rate_high: float


def compute_fed_rate_consensus(
    forecast: MultiSourceFedForecast,
) -> FedRateConsensusResult:
    """
    Determine consensus FOMC rate probabilities from multi-source data.

    For FOMC decisions, the consensus is based on agreement about the
    most likely outcome (hold/cut/hike) and the average probability.
    """
    sources = forecast.sources
    n = len(sources)

    if n == 0:
        return FedRateConsensusResult(
            hold_probability=0.0,
            cut_probability=0.0,
            hike_probability=0.0,
            confidence="skip",
            agreement_ratio=0.0,
            source_count=0,
            meeting_date=forecast.meeting_date,
            current_rate_low=4.25,
            current_rate_high=4.50,
        )

    if n == 1:
        s = sources[0]
        return FedRateConsensusResult(
            hold_probability=s.hold_probability,
            cut_probability=s.cut_probability,
            hike_probability=s.hike_probability,
            confidence="medium",
            agreement_ratio=1.0,
            source_count=1,
            meeting_date=forecast.meeting_date,
            current_rate_low=s.current_rate_low,
            current_rate_high=s.current_rate_high,
        )

    # Determine what each source thinks the most likely outcome is
    def _most_likely(s: FedRateForecast) -> str:
        probs = {"hold": s.hold_probability, "cut": s.cut_probability, "hike": s.hike_probability}
        return max(probs, key=probs.get)

    votes = [_most_likely(s) for s in sources]
    from collections import Counter
    vote_counts = Counter(votes)
    majority_outcome, majority_count = vote_counts.most_common(1)[0]

    agreement_ratio = majority_count / n

    # Average probabilities across sources
    avg_hold = sum(s.hold_probability for s in sources) / n
    avg_cut = sum(s.cut_probability for s in sources) / n
    avg_hike = sum(s.hike_probability for s in sources) / n

    # Normalize
    total = avg_hold + avg_cut + avg_hike
    if total > 0:
        avg_hold /= total
        avg_cut /= total
        avg_hike /= total

    # Confidence based on agreement
    if agreement_ratio >= 0.80:
        confidence = "high"
    elif agreement_ratio >= 0.60:
        confidence = "medium"
    else:
        confidence = "low"

    # Use median current rate
    current_rate_low = sorted(s.current_rate_low for s in sources)[n // 2]
    current_rate_high = sorted(s.current_rate_high for s in sources)[n // 2]

    return FedRateConsensusResult(
        hold_probability=avg_hold,
        cut_probability=avg_cut,
        hike_probability=avg_hike,
        confidence=confidence,
        agreement_ratio=agreement_ratio,
        source_count=n,
        meeting_date=forecast.meeting_date,
        current_rate_low=current_rate_low,
        current_rate_high=current_rate_high,
    )


# ============================================================
# Market discovery + bracket parsing
# ============================================================

# Kalshi series tickers for FOMC rate decisions
FED_RATE_SERIES = [
    "KXFEDDECISION", "FEDDECISION", "KXFED", "FED",
    "KXFOMC", "FOMC", "FEDRATE", "KXFEDRATE",
    "RATECUTCOUNT", "KXRATECUTCOUNT",
]


async def _discover_fed_markets(
    kalshi_client: KalshiClient,
) -> List[TemperatureBracket]:
    """
    Discover active FOMC rate decision markets on Kalshi.
    """
    brackets = []

    for series in FED_RATE_SERIES:
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
                    f"status={m.get('status')}, yes_price={m.get('yes_price')}"
                )

            for market in markets:
                if market.get("status") not in ("active", "open"):
                    continue
                bracket = _parse_fed_bracket(market)
                if bracket:
                    brackets.append(bracket)

            if brackets:
                break

        except Exception as e:
            logger.debug(f"Series '{series}' lookup failed: {e}")
            continue

    logger.info(f"Fed rate markets: {len(brackets)} active brackets discovered")
    return brackets


def _parse_fed_bracket(market: dict) -> Optional[TemperatureBracket]:
    """
    Parse a Fed rate decision bracket from a Kalshi market.

    Expected title formats:
      "Fed maintains rate" / "Fed holds rate"
      "Fed cuts rate by 25bps" / "Fed lowers rate"
      "Fed hikes rate by 25bps" / "Fed raises rate"
      "Rate cut at March meeting"
      "4.00%-4.25%" (rate range brackets)
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

    # Try structured API fields first
    floor_strike = market.get("floor_strike")
    cap_strike = market.get("cap_strike")
    if floor_strike is not None or cap_strike is not None:
        # Rate brackets: floor_strike and cap_strike are in percentage points
        if floor_strike is not None:
            low = int(round(float(floor_strike) * 100))  # Convert to basis points
        if cap_strike is not None:
            high = int(round(float(cap_strike) * 100))

    # Fall through to regex if structured fields didn't work
    if low is None and high is None:
        title_lower = title.lower()

        # Pattern: rate range "4.00%-4.25%" or "4.00% to 4.25%"
        range_match = re.search(r'(\d+\.?\d*)\s*%?\s*[-–to]+\s*(\d+\.?\d*)\s*%', title)
        if range_match:
            low = int(round(float(range_match.group(1)) * 100))
            high = int(round(float(range_match.group(2)) * 100))

        # Pattern: "maintains" / "holds" → hold bracket (change = 0bps)
        elif re.search(r'maintain|hold|no\s*change|unchanged', title_lower):
            # Use 0 as a sentinel for "hold" — will be mapped to current rate
            low = 0
            high = 0  # Special: both 0 means "hold"

        # Pattern: "cuts by 25" or "lower by 25"
        elif re.search(r'cut|lower|decrease|reduc', title_lower):
            bps_match = re.search(r'(\d+)\s*(?:bps|basis)', title_lower)
            if bps_match:
                change = -int(bps_match.group(1))
            else:
                change = -25  # Default cut size
            low = change
            high = change  # Special: equal values means specific change

        # Pattern: "hikes by 25" or "raises by 25"
        elif re.search(r'hike|raise|increase', title_lower):
            bps_match = re.search(r'(\d+)\s*(?:bps|basis)', title_lower)
            if bps_match:
                change = int(bps_match.group(1))
            else:
                change = 25  # Default hike size
            low = change
            high = change

        # Pattern: number of cuts "at least 1 cut" / "2 or more cuts"
        elif re.search(r'(\d+)\s*(?:or more|at least|\+)?\s*cut', title_lower):
            num_match = re.search(r'(\d+)\s*(?:or more|at least|\+)?\s*cut', title_lower)
            if num_match:
                num_cuts = int(num_match.group(1))
                low = -(num_cuts * 25)
                high = None  # "at least X cuts" = open-ended below

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
# Probability mapping: consensus → bracket probabilities
# ============================================================

def _map_consensus_to_bracket_probs(
    consensus: FedRateConsensusResult,
    brackets: List[TemperatureBracket],
) -> Dict[str, float]:
    """
    Map consensus hold/cut/hike probabilities to individual bracket probabilities.

    This is different from the Gaussian model used for weather/econ because
    FOMC decisions are discrete events (hold, cut 25, cut 50, hike 25, etc.)
    rather than continuous distributions.
    """
    bracket_probs = {}

    for bracket in brackets:
        ticker = bracket.ticker
        low = bracket.low
        high = bracket.high

        if low is None and high is None:
            bracket_probs[ticker] = 0.0
            continue

        # Special case: hold bracket (both 0)
        if low == 0 and high == 0:
            bracket_probs[ticker] = consensus.hold_probability
            continue

        # Special case: specific change bracket (low == high)
        if low is not None and high is not None and low == high:
            change_bps = low
            if change_bps == 0:
                bracket_probs[ticker] = consensus.hold_probability
            elif change_bps == -25:
                bracket_probs[ticker] = consensus.cut_probability * 0.85  # Most cuts are 25bps
            elif change_bps == -50:
                bracket_probs[ticker] = consensus.cut_probability * 0.15
            elif change_bps == 25:
                bracket_probs[ticker] = consensus.hike_probability * 0.85
            elif change_bps == 50:
                bracket_probs[ticker] = consensus.hike_probability * 0.15
            else:
                bracket_probs[ticker] = 0.01
            continue

        # Rate range brackets (e.g., 400-425 means 4.00%-4.25%)
        if low is not None and high is not None:
            bracket_mid = (low + high) / 2.0
            current_mid = (consensus.current_rate_low + consensus.current_rate_high) / 2.0 * 100

            diff_bps = bracket_mid - current_mid

            if abs(diff_bps) < 13:  # Within current range → hold
                bracket_probs[ticker] = consensus.hold_probability
            elif diff_bps < -13:  # Below current → cut
                cuts_needed = abs(diff_bps) / 25
                if cuts_needed <= 1:
                    bracket_probs[ticker] = consensus.cut_probability * 0.85
                elif cuts_needed <= 2:
                    bracket_probs[ticker] = consensus.cut_probability * 0.15
                else:
                    bracket_probs[ticker] = 0.01
            else:  # Above current → hike
                hikes_needed = diff_bps / 25
                if hikes_needed <= 1:
                    bracket_probs[ticker] = consensus.hike_probability * 0.85
                elif hikes_needed <= 2:
                    bracket_probs[ticker] = consensus.hike_probability * 0.15
                else:
                    bracket_probs[ticker] = 0.01
            continue

        # Open-ended brackets (e.g., "at least 1 cut")
        if low is not None and high is None:
            # "at least X bps of cuts"
            if low < 0:
                bracket_probs[ticker] = consensus.cut_probability
            else:
                bracket_probs[ticker] = consensus.hike_probability
            continue

        if high is not None and low is None:
            if high > 0:
                bracket_probs[ticker] = consensus.hold_probability + consensus.cut_probability
            else:
                bracket_probs[ticker] = consensus.hike_probability
            continue

        bracket_probs[ticker] = 0.0

    return bracket_probs


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
                f"FED RATE auto-switch: {len(rows)} settled, "
                f"win_rate={win_rate:.1f}%, pnl=${total_pnl:.2f} — switching to LIVE"
            )
            return True
        return False
    except Exception:
        return False


# ============================================================
# Main FOMC rate consensus trading cycle
# ============================================================

async def run_fed_rate_consensus_cycle(
    kalshi_client: KalshiClient,
    db_manager: DatabaseManager,
    paper_mode: bool = False,
) -> Dict:
    """
    Run one complete FOMC rate decision consensus trading cycle.

    1. Check if we're close enough to an FOMC meeting to trade
    2. Fetch rate probabilities from 3 sources
    3. Compute consensus
    4. Discover Kalshi FOMC markets
    5. Map consensus probabilities to bracket probabilities
    6. Generate trade signals where consensus diverges from market prices
    7. Execute trades (paper or live)
    """
    strategy_tag = "fed_rate_consensus"
    logger.info(f"FED RATE CONSENSUS: Starting cycle (paper={paper_mode})...")

    results: Dict = {
        "brackets_found": 0,
        "signals_generated": 0,
        "orders_placed": 0,
        "total_position_value": 0.0,
        "paper_mode": paper_mode,
    }

    # Check if we're close enough to an FOMC meeting
    days_to_meeting = get_days_to_next_meeting()
    next_meeting = get_next_fomc_meeting()

    if days_to_meeting > 45:
        logger.info(
            f"FED RATE CONSENSUS: Next meeting {next_meeting} is {days_to_meeting} days away — "
            f"too far out, skipping"
        )
        return results

    logger.info(
        f"FED RATE CONSENSUS: Next FOMC meeting {next_meeting} in {days_to_meeting} days"
    )

    # 1. Fetch rate forecasts from 3 sources
    forecast = await fetch_fed_rate_forecasts()
    if not forecast.sources:
        logger.warning("FED RATE CONSENSUS: No sources available")
        return results

    # 2. Compute consensus
    consensus = compute_fed_rate_consensus(forecast)

    logger.info(
        f"FED RATE CONSENSUS: confidence={consensus.confidence}, "
        f"hold={consensus.hold_probability:.1%}, cut={consensus.cut_probability:.1%}, "
        f"hike={consensus.hike_probability:.1%}, "
        f"agree={consensus.agreement_ratio:.0%}, sources={consensus.source_count}"
    )

    if consensus.confidence == "skip":
        logger.info("FED RATE CONSENSUS: skipping (insufficient sources)")
        return results

    # 3. Discover Kalshi FOMC markets
    brackets = await _discover_fed_markets(kalshi_client)
    results["brackets_found"] = len(brackets)

    if not brackets:
        logger.info("FED RATE CONSENSUS: no active brackets found")
        return results

    # 4. Map consensus to bracket probabilities
    bracket_probs = _map_consensus_to_bracket_probs(consensus, brackets)

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
            logger.warning(f"FED RATE CONSENSUS: Insufficient bankroll: ${bankroll:.2f}")
            return results

    # 6. Generate trade signals
    # For Fed rate, we compare our consensus probability against market price
    all_signals: List[WeatherTradeSignal] = []

    for bracket in brackets:
        ticker = bracket.ticker
        our_prob = bracket_probs.get(ticker, 0.0)
        market_yes_price = bracket.yes_price / 100.0 if bracket.yes_price else 0.5
        market_no_price = bracket.no_price / 100.0 if bracket.no_price else 0.5

        # Calculate edge
        yes_edge = our_prob - market_yes_price
        no_edge = (1 - our_prob) - market_no_price

        min_edge = 0.08  # 8% minimum edge

        if yes_edge > min_edge:
            # Buy YES — we think this outcome is more likely than market prices
            position_size = min(bankroll * 0.05, 5.0)  # Max $5 per bracket
            signal = WeatherTradeSignal(
                bracket=bracket,
                our_prob=our_prob,
                market_prob=market_yes_price,
                edge=yes_edge,
                edge_pct=yes_edge * 100,
                side="yes",
                confidence=consensus.confidence,
                limit_price=bracket.yes_ask or bracket.yes_price,
                shares=min(5, max(1, int(position_size / (bracket.yes_ask / 100.0)))) if bracket.yes_ask else 1,
                position_size_dollars=position_size,
                city="FED",
                rationale=(
                    f"FED-RATE({consensus.confidence}): "
                    f"consensus={our_prob:.0%} vs market={market_yes_price:.0%}, "
                    f"edge={yes_edge:.0%}, meeting={consensus.meeting_date}"
                ),
            )
            all_signals.append(signal)

        elif no_edge > min_edge:
            # Buy NO — we think this outcome is less likely than market prices
            position_size = min(bankroll * 0.05, 5.0)
            signal = WeatherTradeSignal(
                bracket=bracket,
                our_prob=1 - our_prob,
                market_prob=market_no_price,
                edge=no_edge,
                edge_pct=no_edge * 100,
                side="no",
                confidence=consensus.confidence,
                limit_price=bracket.no_ask or bracket.no_price,
                shares=min(5, max(1, int(position_size / (bracket.no_ask / 100.0)))) if bracket.no_ask else 1,
                position_size_dollars=position_size,
                city="FED",
                rationale=(
                    f"FED-RATE({consensus.confidence}): "
                    f"consensus_no={1-our_prob:.0%} vs market_no={market_no_price:.0%}, "
                    f"edge={no_edge:.0%}, meeting={consensus.meeting_date}"
                ),
            )
            all_signals.append(signal)

    results["signals_generated"] = len(all_signals)

    if not all_signals:
        logger.info("FED RATE CONSENSUS: No trade signals generated this cycle")
        return results

    # Sort by edge, take top opportunities
    all_signals.sort(key=lambda s: s.edge, reverse=True)
    max_trades = 3  # Conservative for new strategy

    logger.info(f"FED RATE CONSENSUS: Top signals ({len(all_signals)} total):")
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
                f"FED RATE PAPER TRADE: {signal.bracket.ticker} {signal.side} "
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
        f"FED RATE CONSENSUS cycle complete: {results['brackets_found']} brackets, "
        f"{results['signals_generated']} signals, {results['orders_placed']} orders "
        f"({'PAPER' if paper_mode else 'LIVE'})"
    )
    return results
