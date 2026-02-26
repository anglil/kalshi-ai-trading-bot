"""
Weather Trading Strategy for Kalshi

Exploits mispricing in Kalshi's daily temperature markets by comparing
NWS weather forecasts to market prices. Uses a Gaussian error model to
convert point forecasts into bracket probabilities, then places limit
orders on brackets where our estimated probability exceeds the market
price by a significant edge.

Key concepts:
- NWS forecasts have historical error σ ≈ 2-3°F for same-day, 4-5°F for next-day
- Kalshi temperature brackets are 2°F wide with edge brackets for tails
- Settlement is based on NWS Daily Climate Report (next morning)
- We use limit orders to minimize fees (taker fee = 0.07 * P * (1-P))
"""

import asyncio
import math
import uuid
from dataclasses import dataclass, field
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
    sigma: float = 3.0
) -> Dict[str, float]:
    """
    Convert a point forecast into bracket probabilities using a Gaussian
    error model.
    
    The NWS point forecast is treated as the mean of a normal distribution
    with standard deviation σ (historical forecast error). The probability
    of each bracket is the integral of the normal PDF over the bracket range.
    
    Args:
        forecast_high: NWS forecasted high temperature (°F)
        brackets: List of temperature brackets
        sigma: Forecast error std deviation (°F). Typical values:
               - Same-day morning: 2.0°F
               - Same-day evening before: 3.0°F
               - Next-day: 4.0°F
    
    Returns:
        Dict mapping bracket ticker to probability
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
            # Middle bracket (e.g. 41-42°)
            z_high = (bracket.high + 0.5 - forecast_high) / sigma
            z_low = (bracket.low - 0.5 - forecast_high) / sigma
            probs[bracket.ticker] = _normal_cdf(z_high) - _normal_cdf(z_low)
        else:
            probs[bracket.ticker] = 0.0
    
    # Normalize to ensure probabilities sum to 1
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
    
    Searches Kalshi for markets matching the city's series ticker
    and target date, then parses bracket ranges from market titles.
    """
    brackets = []
    
    try:
        # Search for weather markets using series ticker
        # Kalshi weather tickers follow patterns like: KXHIGHNY-26FEB22
        markets_response = await kalshi_client.get_markets(
            limit=100,
            series_ticker=station.kalshi_series
        )
        
        markets = markets_response.get("markets", [])
        if not markets:
            # Try alternate ticker patterns
            # Some cities use KXTEMP or HIGHTEMP prefixes
            for prefix in [station.kalshi_series, f"KXTEMP{station.station_id[1:]}", f"HIGHTEMP"]:
                markets_response = await kalshi_client.get_markets(
                    limit=100,
                    series_ticker=prefix
                )
                markets = markets_response.get("markets", [])
                if markets:
                    break
        
        logger.info(f"🔍 Found {len(markets)} weather markets for {station.city}")
        
        for market in markets:
            ticker = market.get("ticker", "")
            title = market.get("title", "")
            
            # Only consider active markets
            if market.get("status") != "active":
                continue
                
            # Parse bracket from title like "Highest temperature in NYC 39°F to 40°F?"
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
    
    # Extract YES/NO prices — skip bracket if no real price data
    yes_price = market.get("yes_price") or market.get("yes_bid") or market.get("yes_ask")
    no_price = market.get("no_price") or market.get("no_bid") or market.get("no_ask")

    if yes_price is None and no_price is None:
        return None  # No price data — cannot assess edge

    # Fill in the missing side from the other
    if yes_price is not None and no_price is None:
        no_price = 100 - yes_price
    elif no_price is not None and yes_price is None:
        yes_price = 100 - no_price

    yes_ask = market.get("yes_ask") or yes_price
    no_ask = market.get("no_ask") or no_price
    volume = market.get("volume", 0)
    
    # Parse temperature range from title
    # Formats for Kalshi:
    # "Will the high temp in Chicago be >33° on Feb 23, 2026?"
    # "Will the high temp in Chicago be <26° on Feb 23, 2026?"
    # "Will the high temp in Chicago be 32-33° on Feb 23, 2026?"
    
    low = None
    high = None
    
    # Try >X°
    gt_match = re.search(r'>(\d+)°', title)
    if gt_match:
        low = int(gt_match.group(1)) + 1 # >40 means 41 or higher (assuming integers)
    else:
        # Try <X°
        lt_match = re.search(r'<(\d+)°', title)
        if lt_match:
            high = int(lt_match.group(1)) - 1 # <33 means 32 or lower
        else:
            # Try X-Y°
            range_match = re.search(r'(\d+)-(\d+)°', title)
            if range_match:
                low = int(range_match.group(1))
                high = int(range_match.group(2))
    
    if low is None and high is None:
        # Couldn't parse bracket — skip
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
# Signal generation
# ============================================================

def generate_weather_signals(
    brackets: List[TemperatureBracket],
    bracket_probs: Dict[str, float],
    city: str,
    bankroll: float,
    min_edge: float = 0.08,
    max_position_pct: float = 0.05,
    kelly_fraction: float = 0.5,
    rationale_prefix: str = "WEATHER",
) -> List[WeatherTradeSignal]:
    """
    Generate trade signals by comparing our probability estimates
    to market prices.
    
    Args:
        brackets: List of temperature brackets with market prices
        bracket_probs: Our probability estimates for each bracket
        city: City name for logging
        bankroll: Total available capital
        min_edge: Minimum edge to trade (default 8%)
        max_position_pct: Max position as % of bankroll
        kelly_fraction: Fraction of full Kelly to use
    
    Returns:
        List of trade signals sorted by edge (strongest first)
    """
    signals = []
    
    for bracket in brackets:
        our_prob = bracket_probs.get(bracket.ticker, 0.0)
        market_prob = bracket.implied_prob
        
        # Skip brackets with no data
        if our_prob <= 0.01 or market_prob <= 0.01 or market_prob >= 0.99:
            continue
        
        # Calculate edge in both directions
        yes_edge = our_prob - market_prob      # Edge for buying YES
        no_edge = market_prob - our_prob        # Edge for buying NO
        
        # Determine best side
        if yes_edge >= min_edge:
            side = "YES"
            edge = yes_edge
            # For YES: we pay market_prob, win (1 - market_prob) if correct
            odds = (1 - market_prob) / market_prob if market_prob > 0 else 0
            win_prob = our_prob
        elif no_edge >= min_edge:
            side = "NO"
            edge = no_edge
            # For NO: we pay (1 - market_prob), win market_prob if correct
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
        
        if net_edge < 0.03:  # Skip if edge doesn't cover fees + 3% margin
            continue
        
        # Minimum position: at least 1 contract
        entry_price_dollars = entry_price_cents / 100.0
        if entry_price_dollars <= 0:
            continue
        
        shares = max(1, int(position_size / entry_price_dollars))
        shares = min(shares, 10)  # cap to prevent penny bet accumulation
        actual_position = shares * entry_price_dollars
        
        # Set limit price: our fair value minus some margin for fill
        if side == "YES":
            # We want to buy YES at a good price (below fair value)
            limit_price = min(entry_price_cents, int(our_prob * 100) - 1)
            limit_price = max(1, limit_price)
        else:
            # We want to buy NO at a good price
            limit_price = min(entry_price_cents, int((1 - our_prob) * 100) - 1)
            limit_price = max(1, limit_price)
        
        # Build bracket description
        if bracket.low is None:
            bracket_desc = f"≤{bracket.high}°F"
        elif bracket.high is None:
            bracket_desc = f"≥{bracket.low}°F"
        else:
            bracket_desc = f"{bracket.low}-{bracket.high}°F"
        
        signal = WeatherTradeSignal(
            bracket=bracket,
            our_prob=our_prob,
            market_prob=market_prob,
            edge=edge,
            edge_pct=edge,
            side=side,
            confidence=min(1.0, our_prob + 0.1),  # Higher confidence when we have strong model signal
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
        # Check for existing position BEFORE placing order to prevent stacking
        existing = await db_manager.get_position_by_market_and_side(
            signal.bracket.ticker, signal.side,
        )
        if existing:
            logger.info(
                f"⏭️ SKIP: Already hold {signal.bracket.ticker} {signal.side} — "
                f"no duplicate order placed"
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

        # Set limit price
        if side_lower == "yes":
            order_kwargs["yes_price"] = signal.limit_price
        else:
            order_kwargs["no_price"] = signal.limit_price

        logger.info(
            f"🌡️ WEATHER TRADE: {signal.city} {signal.bracket.ticker} — "
            f"{signal.shares} {signal.side} @ {signal.limit_price}¢ "
            f"(edge: {signal.edge:.0%})"
        )

        order_response = await kalshi_client.place_order(**order_kwargs)
        
        if order_response and "order" in order_response:
            order_id = order_response["order"].get("order_id", client_order_id)
            
            # Record position in database
            # Weather positions should hold to settlement — set wide exit levels
            # to prevent the tracking job from closing them on intraday noise.
            position = Position(
                market_id=signal.bracket.ticker,
                side=signal.side,
                quantity=signal.shares,
                entry_price=signal.limit_price / 100.0,
                live=True,
                timestamp=datetime.now(),
                rationale=signal.rationale,
                strategy=strategy,
                stop_loss_price=0.01,   # effectively disabled — hold to settlement
                take_profit_price=0.99, # effectively disabled — hold to settlement
                max_hold_hours=48,  # weather markets settle within 24-48h
            )
            await db_manager.add_position(position)
            
            logger.info(
                f"✅ WEATHER ORDER PLACED: {signal.bracket.ticker} — "
                f"Order ID: {order_id}, {signal.shares} {signal.side} @ {signal.limit_price}¢"
            )
            return True
        else:
            logger.error(f"❌ Weather order failed: {order_response}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error executing weather trade: {e}")
        return False


# ============================================================
# Main trading cycle
# ============================================================

async def run_weather_trading_cycle(
    kalshi_client: KalshiClient,
    db_manager: DatabaseManager,
) -> Dict:
    """
    Run one complete weather trading cycle:
    1. Fetch NWS forecasts for all cities
    2. Discover Kalshi weather markets
    3. Calculate bracket probabilities
    4. Find mispriced brackets
    5. Place limit orders
    
    Returns summary dict with results.
    """
    logger.info("🌤️ Starting weather trading cycle...")
    
    results = {
        "cities_analyzed": 0,
        "brackets_found": 0,
        "signals_generated": 0,
        "orders_placed": 0,
        "total_position_value": 0.0,
    }
    
    # Get available balance
    try:
        balance_response = await kalshi_client.get_balance()
        bankroll = balance_response.get("balance", 0) / 100.0  # cents → dollars
    except Exception as e:
        logger.error(f"Could not fetch balance: {e}")
        return results
    
    # Use full available balance — no new deposits, trade only with what's in the account
    weather_bankroll = bankroll
    
    if weather_bankroll < 5.0:
        logger.warning(f"⚠️ Insufficient weather bankroll: ${weather_bankroll:.2f}")
        return results
    
    # Target dates: today and tomorrow
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    all_signals: List[WeatherTradeSignal] = []
    
    for city_key, station in WEATHER_STATIONS.items():
        try:
            results["cities_analyzed"] += 1
            
            # 1. Fetch NWS forecast
            forecast_periods = await get_hourly_forecast(station)
            if not forecast_periods:
                logger.warning(f"⚠️ No forecast data for {station.city}")
                continue
            
            # 2. Get forecasted high for today
            forecast_high = get_forecast_high_for_date(forecast_periods, today)
            if forecast_high is None:
                # Try tomorrow
                forecast_high = get_forecast_high_for_date(forecast_periods, tomorrow)
                if forecast_high is None:
                    logger.warning(f"⚠️ No high temp forecast for {station.city}")
                    continue
            
            logger.info(f"🌡️ {station.city}: NWS forecast high = {forecast_high}°F")
            
            # 3. Discover Kalshi weather markets for this city
            brackets = await discover_weather_markets(kalshi_client, station, today)
            results["brackets_found"] += len(brackets)
            
            if not brackets:
                logger.info(f"📭 No active weather brackets for {station.city}")
                continue
            
            logger.info(f"📊 {station.city}: {len(brackets)} active brackets")
            
            # 4. Calculate bracket probabilities
            # Use tighter σ for same-day (more confident), wider for next-day
            current_hour = datetime.now().hour
            if current_hour >= 10:  # After 10 AM, same-day forecasts are more accurate
                sigma = 2.0
            elif current_hour >= 6:
                sigma = 2.5
            else:
                sigma = 3.5  # Early morning — less certain
            
            bracket_probs = forecast_to_bracket_probs(forecast_high, brackets, sigma=sigma)
            
            # Log probabilities
            for bracket in brackets:
                prob = bracket_probs.get(bracket.ticker, 0)
                if bracket.low is None:
                    desc = f"≤{bracket.high}°F"
                elif bracket.high is None:
                    desc = f"≥{bracket.low}°F"
                else:
                    desc = f"{bracket.low}-{bracket.high}°F"
                logger.debug(
                    f"  {desc}: NWS={prob:.1%}, Market={bracket.implied_prob:.1%}, "
                    f"Edge={prob - bracket.implied_prob:+.1%}"
                )
            
            # 5. Generate trade signals
            city_signals = generate_weather_signals(
                brackets=brackets,
                bracket_probs=bracket_probs,
                city=station.city,
                bankroll=weather_bankroll,
                min_edge=0.08,          # 8% minimum edge
                max_position_pct=0.05,  # 5% per bracket
                kelly_fraction=0.5,     # Half Kelly
            )
            
            if city_signals:
                logger.info(
                    f"🎯 {station.city}: {len(city_signals)} trade signals "
                    f"(best edge: {city_signals[0].edge:.0%})"
                )
                all_signals.extend(city_signals)
            
            # Small delay between cities
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error analyzing {station.city}: {e}")
            continue
    
    results["signals_generated"] = len(all_signals)
    
    if not all_signals:
        logger.info("📊 No weather trade signals generated this cycle")
        return results
    
    # Sort all signals by edge and take top opportunities
    all_signals.sort(key=lambda s: s.edge, reverse=True)
    max_trades = 5  # Max trades per cycle
    
    logger.info(f"🌡️ Top weather signals ({len(all_signals)} total):")
    for i, sig in enumerate(all_signals[:max_trades]):
        logger.info(f"  #{i+1}: {sig.rationale}")
    
    # 6. Execute top signals
    for signal in all_signals[:max_trades]:
        success = await execute_weather_trade(signal, kalshi_client, db_manager)
        if success:
            results["orders_placed"] += 1
            results["total_position_value"] += signal.position_size_dollars
    
    logger.info(
        f"🌤️ Weather cycle complete: {results['cities_analyzed']} cities, "
        f"{results['brackets_found']} brackets, {results['signals_generated']} signals, "
        f"{results['orders_placed']} orders placed (${results['total_position_value']:.2f})"
    )
    
    return results
