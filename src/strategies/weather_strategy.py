"""
Weather Trading Strategy for Kalshi

Exploits mispricing in Kalshi's daily temperature markets by comparing
NWS weather forecasts to market prices. Uses a Gaussian error model to
convert point forecasts into bracket probabilities, then places limit
orders on brackets where our estimated probability exceeds the market
price by a significant edge.

Key concepts:
- NWS forecasts have historical error sigma 5-7 deg F (calibrated from real losses)
- Kalshi temperature brackets are 2 deg F wide with edge brackets for tails
- Settlement is based on NWS Daily Climate Report (next morning)
- We use limit orders to minimize fees (taker fee = 0.07 * P * (1-P))

TURNAROUND v3: Only bet on the 1-2 brackets CLOSEST to the forecast.
Historical data shows the big wins came from "obvious" brackets (e.g.
Chicago >40F in Feb, Miami >65F) bought at 10-50c. The losses came from
spreading capital across 10+ brackets, most of which expired worthless.
"""

import asyncio
import math
import uuid
from dataclasses import dataclass, field
from src.jobs.execute import _check_existing_kalshi_position
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from src.clients.nws_client import (
    WEATHER_STATIONS,
    WeatherStation,
    get_hourly_forecast,
    get_forecast_high_for_date,
    get_forecast_temps_for_date,
    get_current_observation,
)
from src.clients.kalshi_client import KalshiClient
from src.utils.database import DatabaseManager, Position
from src.utils.logging_setup import get_trading_logger
from src.config.settings import settings

logger = get_trading_logger("weather_strategy")


# ============================================================
# Data structures
# ============================================================

@dataclass
class TemperatureBracket:
    """A single Kalshi temperature bracket."""
    ticker: str           # Kalshi market ticker
    low: Optional[int]    # Lower bound (None = "below X")
    high: Optional[int]   # Upper bound (None = "above X")
    yes_price: int        # Current YES price in cents
    no_price: int         # Current NO price in cents
    yes_ask: int          # Best YES ask in cents
    no_ask: int           # Best NO ask in cents
    volume: int           # Trading volume

    @property
    def implied_prob(self) -> float:
        """Market implied probability from YES price."""
        return self.yes_price / 100.0

    @property
    def midpoint(self) -> Optional[float]:
        """Bracket midpoint temperature."""
        if self.low is not None and self.high is not None:
            return (self.low + self.high) / 2.0
        return None

    def contains(self, temp: float) -> bool:
        """Check if a temperature falls in this bracket."""
        if self.low is None:
            return temp <= self.high
        if self.high is None:
            return temp >= self.low
        return self.low <= temp <= self.high


@dataclass
class WeatherTradeSignal:
    """A trade signal for a weather bracket."""
    bracket: TemperatureBracket
    our_prob: float          # Our estimated probability
    market_prob: float       # Market implied probability
    edge: float              # our_prob - market_prob
    edge_pct: float          # edge as percentage
    side: str                # "YES" or "NO"
    confidence: float        # How confident we are (0-1)
    limit_price: int         # Our limit price in cents
    position_size_dollars: float
    shares: int
    city: str
    rationale: str


# ============================================================
# Probability model
# ============================================================

def _normal_cdf(x: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def forecast_to_bracket_probs(
    forecast_high: float,
    brackets: List[TemperatureBracket],
    sigma: float = 7.0,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Convert a point forecast into bracket probabilities using a Gaussian
    error model.
    
    The NWS point forecast is treated as the mean of a normal distribution
    with standard deviation sigma (historical forecast error). The probability
    of each bracket is the integral of the normal PDF over the bracket range.
    
    TURNAROUND v4: Default sigma raised to 7.0 (was 5.0, originally 3.0).
    Historical data showed the model was still overconfident on NO positions,
    leading to systematic losses. Wider sigma = more honest uncertainty.
    """
    probs = {}
    for bracket in brackets:
        if bracket.low is None and bracket.high is not None:
            # "Below X" bracket
            z = (bracket.high + 0.5 - forecast_high) / sigma
            probs[bracket.ticker] = _normal_cdf(z)
        elif bracket.high is None and bracket.low is not None:
            # "Above X" bracket
            z = (bracket.low - 0.5 - forecast_high) / sigma
            probs[bracket.ticker] = 1.0 - _normal_cdf(z)
        elif bracket.low is not None and bracket.high is not None:
            # Middle bracket (e.g. 41-42 deg)
            z_high = (bracket.high + 0.5 - forecast_high) / sigma
            z_low = (bracket.low - 0.5 - forecast_high) / sigma
            probs[bracket.ticker] = _normal_cdf(z_high) - _normal_cdf(z_low)
        else:
            probs[bracket.ticker] = 0.0
    
    # Normalize to ensure probabilities sum to 1
    # SKIP normalization for cumulative brackets (econ markets: "above X%")
    # where each bracket is independent, not exclusive ranges
    if normalize:
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
    
    return probs


def kalshi_taker_fee(price_cents: int) -> float:
    """Calculate Kalshi taker fee for a given price in cents."""
    p = price_cents / 100.0
    return 0.07 * p * (1 - p)


# ============================================================
# Kalshi market discovery
# ============================================================

async def discover_weather_markets(
    kalshi_client: KalshiClient,
    station: WeatherStation,
    target_date: str
) -> List[TemperatureBracket]:
    """
    Discover active temperature bracket markets for a city and date.
    """
    brackets = []
    
    try:
        markets_response = await kalshi_client.get_markets(
            limit=100,
            series_ticker=station.kalshi_series
        )
        
        markets = markets_response.get("markets", [])
        if not markets:
            for prefix in [station.kalshi_series, f"KXTEMP{station.station_id[1:]}", f"HIGHTEMP"]:
                markets_response = await kalshi_client.get_markets(
                    limit=100,
                    series_ticker=prefix
                )
                markets = markets_response.get("markets", [])
                if markets:
                    break
        
        logger.info(f"Found {len(markets)} weather markets for {station.city}")
        
        for market in markets:
            ticker = market.get("ticker", "")
            title = market.get("title", "")
            
            if market.get("status") != "active":
                continue
                
            bracket = _parse_bracket_from_market(market)
            if bracket:
                brackets.append(bracket)

    except Exception as e:
        logger.error(f"Error discovering weather markets for {station.city}: {e}")
    
    return brackets


def _parse_bracket_from_market(market: dict) -> Optional[TemperatureBracket]:
    """Parse a TemperatureBracket from a Kalshi market response."""
    import re
    
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
    
    gt_match = re.search(r'>(\d+)', title)
    if gt_match:
        low = int(gt_match.group(1)) + 1
    else:
        lt_match = re.search(r'<(\d+)', title)
        if lt_match:
            high = int(lt_match.group(1)) - 1
        else:
            range_match = re.search(r'(\d+)-(\d+)', title)
            if range_match:
                low = int(range_match.group(1))
                high = int(range_match.group(2))
    
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
        volume=volume
    )


# ============================================================
# Signal generation — TURNAROUND v3
# ============================================================

# TURNAROUND v3: Minimum entry price raised to 30c.
# Historical data: 0-15c contracts had -57% ROI, 15-30c had -43% ROI.
# Only 30c+ contracts have a chance of being profitable.
MIN_ENTRY_PRICE_CENTS = 30

# FIX #2: Only 1 bracket per city-date — the single most extreme "obvious NO".
# Data shows 3-4 brackets per city = 1 winner + 2-3 losers bleeding cash.
# Picking only the best bracket eliminates the extra losing brackets.
MAX_BRACKETS_PER_CITY = 1


def _bracket_distance_to_forecast(bracket: TemperatureBracket, forecast: float) -> float:
    """
    How far is this bracket from the forecast temperature?
    Lower = closer = more likely to be the winning bracket.
    
    For "above X" brackets: distance = |X - forecast|
    For "below X" brackets: distance = |X - forecast|
    For "X-Y" brackets: distance = |midpoint - forecast|
    """
    if bracket.low is not None and bracket.high is not None:
        midpoint = (bracket.low + bracket.high) / 2.0
        return abs(midpoint - forecast)
    elif bracket.high is None and bracket.low is not None:
        # "Above X" bracket — forecast should be above X for this to win
        return abs(bracket.low - forecast)
    elif bracket.low is None and bracket.high is not None:
        # "Below X" bracket — forecast should be below X for this to win
        return abs(bracket.high - forecast)
    return float('inf')


def generate_weather_signals(
    brackets: List[TemperatureBracket],
    bracket_probs: Dict[str, float],
    city: str,
    bankroll: float,
    min_edge: float = 0.15,
    max_position_pct: float = 0.03,
    kelly_fraction: float = 0.25,
    rationale_prefix: str = "WEATHER",
    max_shares: int = 5,
    forecast_high: float = None,
) -> List[WeatherTradeSignal]:
    """
    Generate trade signals by comparing our probability estimates
    to market prices.
    
    TURNAROUND v3 changes:
    - Only consider the 2 brackets closest to the forecast
    - Minimum entry price raised to 30c (was 15c)
    - Stronger edge requirement (15% minimum)
    - Smaller position sizes (quarter Kelly, 3% max)
    """
    signals = []
    
    # TURNAROUND v3: If we have the forecast, pre-filter to only the
    # closest brackets. This prevents the "spray and pray" approach
    # that lost $248 across dozens of tail brackets.
    if forecast_high is not None:
        # Sort brackets by distance to forecast
        brackets_with_dist = [
            (b, _bracket_distance_to_forecast(b, forecast_high))
            for b in brackets
        ]
        brackets_with_dist.sort(key=lambda x: x[1])
        # Only consider the N closest brackets
        brackets = [b for b, d in brackets_with_dist[:MAX_BRACKETS_PER_CITY * 3]]
        logger.info(
            f"TURNAROUND: {city} — filtered to {len(brackets)} closest brackets "
            f"(forecast={forecast_high:.0f}F)"
        )
    
    for bracket in brackets:
        our_prob = bracket_probs.get(bracket.ticker, 0.0)
        market_prob = bracket.implied_prob
        
        # Skip brackets with no data
        if our_prob <= 0.01 or market_prob <= 0.01 or market_prob >= 0.99:
            continue
        
        # Calculate edge in both directions
        yes_edge = our_prob - market_prob      # Edge for buying YES
        no_edge = market_prob - our_prob        # Edge for buying NO
        
        # Market efficiency check
        if abs(yes_edge) < 0.05 and abs(no_edge) < 0.05:
            continue
        
        # Determine best side
        if yes_edge >= min_edge:
            side = "YES"
            edge = yes_edge
            odds = (1 - market_prob) / market_prob if market_prob > 0 else 0
            win_prob = our_prob
        elif no_edge >= min_edge:
            side = "NO"
            edge = no_edge
            odds = market_prob / (1 - market_prob) if market_prob < 1 else 0
            win_prob = 1 - our_prob
        else:
            continue  # No significant edge
        
        # Kelly criterion position sizing
        if odds > 0:
            kelly = (odds * win_prob - (1 - win_prob)) / odds
            kelly = max(0, kelly) * kelly_fraction
        else:
            kelly = 0
        
        position_size = min(
            kelly * bankroll,
            max_position_pct * bankroll,
        )
        
        # Calculate fee-adjusted edge
        if side == "YES":
            entry_price_cents = bracket.yes_ask if bracket.yes_ask > 0 else bracket.yes_price
        else:
            entry_price_cents = bracket.no_ask if bracket.no_ask > 0 else bracket.no_price
        
        fee = kalshi_taker_fee(entry_price_cents)
        net_edge = edge - fee
        
        if net_edge < 0.05:
            continue
        
        entry_price_dollars = entry_price_cents / 100.0
        if entry_price_dollars <= 0:
            continue
        
        # TURNAROUND v3: Minimum price floor at 30c (was 15c)
        # Historical data: 0-30c contracts have negative ROI across the board
        if entry_price_cents < MIN_ENTRY_PRICE_CENTS:
            logger.debug(
                f"PRICE FLOOR: Skip {bracket.ticker} {side} @ {entry_price_cents}c "
                f"(min {MIN_ENTRY_PRICE_CENTS}c)"
            )
            continue
        
        shares = max(1, int(position_size / entry_price_dollars))
        shares = min(shares, max_shares)
        actual_position = shares * entry_price_dollars
        
        # Set limit price: maker-friendly (2c below ask)
        if side == "YES":
            limit_price = min(entry_price_cents - 2, int(our_prob * 100) - 2)
            limit_price = max(MIN_ENTRY_PRICE_CENTS, limit_price)
        else:
            limit_price = min(entry_price_cents - 2, int((1 - our_prob) * 100) - 2)
            limit_price = max(MIN_ENTRY_PRICE_CENTS, limit_price)
        
        # Reject dead orders — if limit is more than 30% below market ask
        if limit_price < entry_price_cents * 0.50:
            continue
        
        # Build bracket description
        if bracket.low is None:
            bracket_desc = f"<={bracket.high}F"
        elif bracket.high is None:
            bracket_desc = f">={bracket.low}F"
        else:
            bracket_desc = f"{bracket.low}-{bracket.high}F"
        
        signal = WeatherTradeSignal(
            bracket=bracket,
            our_prob=our_prob,
            market_prob=market_prob,
            edge=edge,
            edge_pct=edge,
            side=side,
            confidence=min(1.0, our_prob + 0.1),
            limit_price=limit_price,
            position_size_dollars=actual_position,
            shares=shares,
            city=city,
            rationale=(
                f"{rationale_prefix}: {city} {bracket_desc} — "
                f"NWS prob {our_prob:.0%} vs market {market_prob:.0%} = "
                f"{edge:.0%} edge ({side}), net after fees: {net_edge:.0%}"
            ),
        )
        signals.append(signal)
    
    # Sort by edge (strongest first)
    signals.sort(key=lambda s: s.edge, reverse=True)
    
    # TURNAROUND v3: Only return top MAX_BRACKETS_PER_CITY signals
    if len(signals) > MAX_BRACKETS_PER_CITY:
        logger.info(
            f"TURNAROUND: {city} — trimming from {len(signals)} to "
            f"{MAX_BRACKETS_PER_CITY} signals (top edge only)"
        )
        signals = signals[:MAX_BRACKETS_PER_CITY]
    
    return signals


# ============================================================
# Trade execution
# ============================================================

async def execute_weather_trade(
    signal: WeatherTradeSignal,
    kalshi_client: KalshiClient,
    db_manager: DatabaseManager,
    strategy: str = "weather_forecast",
) -> bool:
    """
    Execute a single weather trade signal using a limit order.
    Returns True if order was placed successfully.
    """
    try:
        # === CAPITAL PROTECTION ===
        # DOUBLE DOWN: Lowered from $50 to $20 for weather (our top earner needs to trade)
        MIN_CASH_TO_TRADE = 20.0
        try:
            bal_resp = await kalshi_client.get_balance()
            available_cash = bal_resp.get('balance', 0) / 100.0
            if available_cash < MIN_CASH_TO_TRADE:
                logger.warning(
                    f"CAPITAL GUARD: Cash ${available_cash:.2f} < ${MIN_CASH_TO_TRADE:.2f} minimum. "
                    f"Refusing to open new position on {signal.bracket.ticker}. "
                    f"Wait for settlements to free up cash."
                )
                return False
        except Exception as e:
            logger.warning(f"Capital check failed: {e}")

        # FIX #3: HARD NO-REBUY RULE
        # If ANY position exists on this ticker (any side, any quantity), refuse.
        # Data shows losers accumulate 3.3x more contracts than winners because
        # the bot keeps re-buying across cycles. One entry per ticker, period.
        try:
            existing_api = await _check_existing_kalshi_position(kalshi_client, signal.bracket.ticker)
            existing_pos = existing_api.get('position', 0)
            existing_qty = abs(existing_pos)
            
            if existing_qty > 0:
                logger.info(
                    f"NO-REBUY: Already hold {existing_qty} contracts on "
                    f"{signal.bracket.ticker} — no re-buying allowed (Fix #3)"
                )
                return False
        except Exception as e:
            logger.warning(f"API position check failed, falling back to DB: {e}")
        
        # Also check local DB as backup
        existing = await db_manager.get_position_by_market_and_side(
            signal.bracket.ticker, signal.side,
        )
        if existing:
            logger.info(
                f"NO-REBUY: Already hold {signal.bracket.ticker} {signal.side} in DB — "
                f"no re-buying allowed (Fix #3)"
            )
            return False

        client_order_id = str(uuid.uuid4())
        side_lower = signal.side.lower()

        order_kwargs = {
            "ticker": signal.bracket.ticker,
            "client_order_id": client_order_id,
            "side": side_lower,
            "action": "buy",
            "count": signal.shares,
            "type_": "limit",
        }

        if side_lower == "yes":
            order_kwargs["yes_price"] = signal.limit_price
        else:
            order_kwargs["no_price"] = signal.limit_price

        logger.info(
            f"WEATHER TRADE: {signal.city} {signal.bracket.ticker} — "
            f"{signal.shares} {signal.side} @ {signal.limit_price}c "
            f"(edge: {signal.edge:.0%})"
        )

        order_response = await kalshi_client.place_order(**order_kwargs)
        
        if order_response and "order" in order_response:
            order_id = order_response["order"].get("order_id", client_order_id)
            
            entry_dollars = signal.limit_price / 100.0
            # Side-aware stop-loss and take-profit levels
            # Take-profit is TIGHT: sell as soon as ~10% profitable (fees are ~3-4%)
            # Stop-loss is moderate: cut losses at ~25% to limit downside
            if signal.side == "YES":
                sl_price = max(0.02, round(entry_dollars * 0.75, 2))   # 25% stop loss
                tp_price = min(0.95, round(entry_dollars * 1.10, 2))   # 10% take profit
            else:  # NO side
                sl_price = min(0.99, round(entry_dollars * 1.25, 2))   # 25% stop loss
                tp_price = max(0.02, round(entry_dollars * 0.90, 2))   # 10% take profit
            position = Position(
                market_id=signal.bracket.ticker,
                side=signal.side,
                quantity=signal.shares,
                entry_price=entry_dollars,
                live=True,
                timestamp=datetime.now(),
                rationale=signal.rationale,
                strategy=strategy,
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
                max_hold_hours=36,
            )
            await db_manager.add_position(position)
            
            logger.info(
                f"WEATHER ORDER PLACED: {signal.bracket.ticker} — "
                f"Order ID: {order_id}, {signal.shares} {signal.side} @ {signal.limit_price}c"
            )
            return True
        else:
            logger.error(f"Weather order failed: {order_response}")
            return False
            
    except Exception as e:
        logger.error(f"Error executing weather trade: {e}")
        return False


# ============================================================
# Main trading cycle
# ============================================================

async def run_weather_trading_cycle(
    kalshi_client: KalshiClient,
    db_manager: DatabaseManager,
) -> Dict:
    """
    Run one complete weather trading cycle.
    
    TURNAROUND v3: Radically simplified.
    - Only bet on 1-2 brackets closest to forecast per city
    - Minimum 30c entry price
    - Wider sigma (5.0+) for honest uncertainty
    - Max 2 trades per cycle total
    """
    logger.info("Starting weather trading cycle (TURNAROUND v3)...")
    
    results = {
        "cities_analyzed": 0,
        "brackets_found": 0,
        "signals_generated": 0,
        "orders_placed": 0,
        "total_position_value": 0.0,
    }
    
    try:
        balance_response = await kalshi_client.get_balance()
        bankroll = balance_response.get("balance", 0) / 100.0
    except Exception as e:
        logger.error(f"Could not fetch balance: {e}")
        return results
    
    # DOUBLE DOWN: Weather gets 50% of portfolio (was 30%) — it's our top earner
    weather_bankroll = bankroll * 0.50
    
    if weather_bankroll < 5.0:
        logger.warning(f"Insufficient weather bankroll: ${weather_bankroll:.2f}")
        return results
    logger.info(f"Weather bankroll: ${weather_bankroll:.2f} (30% of ${bankroll:.2f})")
    
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    all_signals: List[WeatherTradeSignal] = []
    
    for city_key, station in WEATHER_STATIONS.items():
        try:
            results["cities_analyzed"] += 1
            
            forecast_periods = await get_hourly_forecast(station)
            if not forecast_periods:
                logger.warning(f"No forecast data for {station.city}")
                continue
            
            forecast_high = get_forecast_high_for_date(forecast_periods, today)
            if forecast_high is None:
                forecast_high = get_forecast_high_for_date(forecast_periods, tomorrow)
                if forecast_high is None:
                    logger.warning(f"No high temp forecast for {station.city}")
                    continue
            
            logger.info(f"{station.city}: NWS forecast high = {forecast_high}F")
            
            brackets = await discover_weather_markets(kalshi_client, station, today)
            results["brackets_found"] += len(brackets)
            
            if not brackets:
                logger.info(f"No active weather brackets for {station.city}")
                continue
            
            logger.info(f"{station.city}: {len(brackets)} active brackets")
            
            # TURNAROUND v3: Use wider sigma for honest uncertainty
            # Historical NWS error is 3-5F, but our model needs extra buffer
            # because bracket boundaries create cliff effects
            current_hour = datetime.now().hour
            if current_hour >= 10:
                sigma = 6.0   # v4: Was 4.0 — increased to reduce NO-side overconfidence
            elif current_hour >= 6:
                sigma = 7.0   # v4: Was 5.0
            else:
                sigma = 8.0   # v4: Was 6.0
            
            bracket_probs = forecast_to_bracket_probs(forecast_high, brackets, sigma=sigma)
            
            # TURNAROUND v3: Pass forecast_high so signals are filtered to closest brackets
            # DOUBLE DOWN: Bigger bets, lower edge threshold, more shares
            city_signals = generate_weather_signals(
                brackets=brackets,
                bracket_probs=bracket_probs,
                city=station.city,
                bankroll=weather_bankroll,
                min_edge=0.12,
                max_position_pct=0.05,
                kelly_fraction=0.30,
                max_shares=8,
                forecast_high=forecast_high,
            )
            
            if city_signals:
                logger.info(
                    f"{station.city}: {len(city_signals)} trade signals "
                    f"(best edge: {city_signals[0].edge:.0%})"
                )
                all_signals.extend(city_signals)
            
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error analyzing {station.city}: {e}")
            continue
    
    results["signals_generated"] = len(all_signals)
    
    if not all_signals:
        logger.info("No weather trade signals generated this cycle")
        return results
    
    # Sort all signals by edge and take top 2 only
    all_signals.sort(key=lambda s: s.edge, reverse=True)
    max_trades = 4  # 1 bracket per city x 4 cities = max 4 trades per cycle
    
    logger.info(f"Top weather signals ({len(all_signals)} total):")
    for i, sig in enumerate(all_signals[:max_trades]):
        logger.info(f"  #{i+1}: {sig.rationale}")
    
    for signal in all_signals[:max_trades]:
        success = await execute_weather_trade(signal, kalshi_client, db_manager)
        if success:
            results["orders_placed"] += 1
            results["total_position_value"] += signal.position_size_dollars
    
    logger.info(
        f"Weather cycle complete: {results['cities_analyzed']} cities, "
        f"{results['brackets_found']} brackets, {results['signals_generated']} signals, "
        f"{results['orders_placed']} orders placed (${results['total_position_value']:.2f})"
    )
    
    return results
