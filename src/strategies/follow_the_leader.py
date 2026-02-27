"""
Follow the Leader Strategy
==========================

Implements the "Follow the Leader" online learning strategy described in:
  Kalai & Vempala (2005), Hazan et al. (2016), and the UBC MLRG slides
  https://www.cs.ubc.ca/labs/lci/mlrg/slides/2019_summer_3_follow_the_leader.pdf

Core idea: identify the top traders (leaders) on Kalshi by analysing the
public trades feed, then copy their positions on low-frequency markets.

Why low-frequency markets (span ≥ 1 hour)?
  - High-frequency markets (e.g. 15-min crypto price) are too noisy to follow
    reliably — by the time we detect a leader's trade and execute, the edge is gone.
  - Low-frequency markets (weather, economics, politics, sports outrights) have
    enough time for our copy-trade to fill at a reasonable price.

How we identify "leaders":
  Kalshi's leaderboard is not exposed via their public API, so we use the
  public trades feed as a proxy.  Each trade record exposes:
    - ticker, taker_side, count, price, created_time
  We aggregate recent trades (last N minutes) per market and identify
  "smart money" by detecting:
    1. Large single trades (count ≥ LARGE_TRADE_THRESHOLD) — informed bettors
       tend to size up when they have edge.
    2. Persistent directional flow — multiple large trades all on the same side
       within a short window signals conviction.
    3. Price-moving trades — trades that move the market price indicate
       informed order flow rather than noise.

  We rank markets by a "leader signal score" and follow the top 100 signals.

Parameters (all tunable via settings):
  MIN_EVENT_SPAN_HOURS   = 1.0    # only trade markets open for ≥ 1 hour
  LARGE_TRADE_THRESHOLD  = 20     # contracts in a single fill to qualify
  LOOKBACK_MINUTES       = 60     # how far back to scan the trades feed
  TOP_N_LEADERS          = 100    # max number of leader signals to follow
  MIN_LEADER_SCORE       = 3.0    # minimum score to act on a signal
  MAX_POSITION_DOLLARS   = 15.0   # max dollars per follow-trade
  KELLY_FRACTION         = 0.25   # conservative Kelly for copy-trading
  CYCLE_INTERVAL_SECS    = 1800   # run every 30 minutes
"""

import asyncio
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

from src.clients.kalshi_client import KalshiClient
from src.config.settings import settings
from src.utils.database import DatabaseManager, Position
from src.utils.logging_setup import get_trading_logger
from src.strategies.weather_strategy import kalshi_taker_fee

logger = get_trading_logger("follow_the_leader")

# ============================================================
# Configuration
# ============================================================

MIN_EVENT_SPAN_HOURS   = 1.0     # only follow on low-frequency markets
LARGE_TRADE_THRESHOLD  = 20      # min contracts per fill to count as "leader"
LOOKBACK_MINUTES       = 60      # scan last 60 min of public trades
TOP_N_LEADERS          = 100     # follow top 100 leader signals
MIN_LEADER_SCORE       = 3.0     # minimum score to act on
MAX_POSITION_DOLLARS   = 15.0    # max $ per follow-trade
KELLY_FRACTION         = 0.25    # conservative Kelly for copy-trading
MIN_YES_PRICE          = 5       # cents — avoid near-zero markets
MAX_YES_PRICE          = 95      # cents — avoid near-certain markets
MIN_VOLUME             = 50      # min total volume for a market to be liquid enough
CYCLE_INTERVAL_SECS    = 1800    # 30 minutes


# ============================================================
# Data structures
# ============================================================

@dataclass
class LeaderSignal:
    """A detected leader signal on a specific market."""
    ticker: str
    event_ticker: str
    side: str                  # "yes" or "no"
    score: float               # leader signal strength score
    large_trade_count: int     # number of large trades detected
    total_leader_volume: int   # total contracts from large trades
    price_cents: int           # current market price for the side
    market_span_hours: float   # how long the market has been open
    close_time: str            # market close time (ISO)
    rationale: str


@dataclass
class MarketTradeAgg:
    """Aggregated trade data for a single market over the lookback window."""
    ticker: str
    yes_large_count: int = 0
    no_large_count: int = 0
    yes_large_volume: int = 0
    no_large_volume: int = 0
    yes_total_volume: int = 0
    no_total_volume: int = 0
    trade_count: int = 0
    first_price: Optional[int] = None
    last_price: Optional[int] = None


# ============================================================
# Market span filter
# ============================================================

def _compute_span_hours(open_time: str, close_time: str) -> float:
    """Return the total span of a market in hours."""
    try:
        ot = datetime.fromisoformat(open_time.replace('Z', '+00:00'))
        ct = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
        return max(0.0, (ct - ot).total_seconds() / 3600.0)
    except Exception:
        return 0.0


def _is_low_frequency(market: Dict) -> bool:
    """Return True if the market has a span of at least MIN_EVENT_SPAN_HOURS."""
    open_time  = market.get('open_time', '')
    close_time = market.get('close_time', '')
    if not open_time or not close_time:
        return False
    span = _compute_span_hours(open_time, close_time)
    return span >= MIN_EVENT_SPAN_HOURS


# ============================================================
# Public trades aggregation
# ============================================================

async def _fetch_recent_trades(
    kalshi_client: KalshiClient,
    lookback_minutes: int = LOOKBACK_MINUTES,
    max_trades: int = 1000,
) -> List[Dict]:
    """
    Fetch recent public trades across all markets.
    Returns a flat list of trade dicts sorted newest-first.
    """
    try:
        result = await kalshi_client._make_authenticated_request(
            'GET',
            '/trade-api/v2/markets/trades',
            params={'limit': max_trades},
        )
        trades = result.get('trades', [])
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
        recent = []
        for t in trades:
            created = t.get('created_time', '')
            if not created:
                continue
            try:
                ts = datetime.fromisoformat(created.replace('Z', '+00:00'))
                if ts >= cutoff:
                    recent.append(t)
            except Exception:
                continue
        logger.debug(f"FTL: fetched {len(recent)} trades in last {lookback_minutes} min")
        return recent
    except Exception as e:
        logger.warning(f"FTL: failed to fetch public trades: {e}")
        return []


def _aggregate_trades(trades: List[Dict]) -> Dict[str, MarketTradeAgg]:
    """
    Aggregate trades by market ticker.
    Returns a dict of ticker -> MarketTradeAgg.
    """
    agg: Dict[str, MarketTradeAgg] = {}
    for t in trades:
        ticker = t.get('ticker', '')
        if not ticker:
            continue
        if ticker not in agg:
            agg[ticker] = MarketTradeAgg(ticker=ticker)
        a = agg[ticker]
        count = int(t.get('count', 0))
        side  = t.get('taker_side', '').lower()  # 'yes' or 'no'
        price = t.get('yes_price', 0) or 0
        a.trade_count += 1
        if a.first_price is None:
            a.first_price = price
        a.last_price = price
        if side == 'yes':
            a.yes_total_volume += count
            if count >= LARGE_TRADE_THRESHOLD:
                a.yes_large_count  += 1
                a.yes_large_volume += count
        elif side == 'no':
            a.no_total_volume += count
            if count >= LARGE_TRADE_THRESHOLD:
                a.no_large_count  += 1
                a.no_large_volume += count
    return agg


def _score_market(agg: MarketTradeAgg) -> Tuple[str, float]:
    """
    Compute a leader signal score for a market.

    Score formula:
      - Base: max(yes_large_count, no_large_count)
      - Volume bonus: large_volume / 100
      - Directional conviction: ratio of dominant to total large volume
      - Price movement bonus: abs(last_price - first_price) / 10

    Returns (dominant_side, score).  Side is 'yes' or 'no'.
    """
    if agg.yes_large_count == 0 and agg.no_large_count == 0:
        return ('yes', 0.0)

    if agg.yes_large_volume >= agg.no_large_volume:
        dominant_side   = 'yes'
        dom_large_count = agg.yes_large_count
        dom_large_vol   = agg.yes_large_volume
        sub_large_vol   = agg.no_large_volume
    else:
        dominant_side   = 'no'
        dom_large_count = agg.no_large_count
        dom_large_vol   = agg.no_large_volume
        sub_large_vol   = agg.yes_large_volume

    total_large_vol = dom_large_vol + sub_large_vol
    conviction = dom_large_vol / total_large_vol if total_large_vol > 0 else 0.5

    # Price movement (leaders moving the price = informed)
    price_move = 0.0
    if agg.first_price is not None and agg.last_price is not None:
        price_move = abs(agg.last_price - agg.first_price) / 10.0

    score = (
        dom_large_count * 1.0
        + dom_large_vol / 100.0
        + conviction * 2.0
        + price_move
    )

    return (dominant_side, round(score, 3))


# ============================================================
# Market detail enrichment
# ============================================================

async def _enrich_with_market_details(
    kalshi_client: KalshiClient,
    tickers: List[str],
) -> Dict[str, Dict]:
    """
    Fetch market details for a list of tickers.
    Returns dict of ticker -> market dict.
    """
    details: Dict[str, Dict] = {}
    for ticker in tickers:
        try:
            result = await kalshi_client.get_market(ticker)
            market = result.get('market', result)
            details[ticker] = market
        except Exception as e:
            logger.debug(f"FTL: could not fetch market {ticker}: {e}")
    return details


# ============================================================
# Signal generation
# ============================================================

async def _generate_leader_signals(
    kalshi_client: KalshiClient,
    lookback_minutes: int = LOOKBACK_MINUTES,
) -> List[LeaderSignal]:
    """
    Scan the public trades feed and generate Follow the Leader signals.

    Steps:
    1. Fetch recent public trades
    2. Aggregate by market
    3. Score each market
    4. Filter to low-frequency markets with strong leader signal
    5. Enrich with market details (price, span, etc.)
    6. Return top TOP_N_LEADERS signals sorted by score
    """
    trades = await _fetch_recent_trades(kalshi_client, lookback_minutes)
    if not trades:
        logger.info("FTL: no recent trades found")
        return []

    agg = _aggregate_trades(trades)
    logger.info(f"FTL: aggregated {len(agg)} markets from {len(trades)} trades")

    # Score all markets
    scored: List[Tuple[str, str, float, MarketTradeAgg]] = []
    for ticker, a in agg.items():
        side, score = _score_market(a)
        if score >= MIN_LEADER_SCORE:
            scored.append((ticker, side, score, a))

    # Sort by score descending
    scored.sort(key=lambda x: -x[2])
    top_scored = scored[:TOP_N_LEADERS * 3]  # fetch more than needed, filter after

    if not top_scored:
        logger.info("FTL: no markets met the minimum leader score threshold")
        return []

    # Enrich with market details
    tickers_to_enrich = [t[0] for t in top_scored]
    details = await _enrich_with_market_details(kalshi_client, tickers_to_enrich)

    signals: List[LeaderSignal] = []
    for ticker, side, score, a in top_scored:
        market = details.get(ticker)
        if not market:
            continue

        # Filter: must be low-frequency
        if not _is_low_frequency(market):
            logger.debug(f"FTL: skip {ticker} — high-frequency market")
            continue

        # Filter: must be open/active
        status = market.get('status', '')
        if status not in ('active', 'open', ''):
            continue

        # Filter: price must be in reasonable range
        yes_price = market.get('yes_price') or market.get('yes_bid') or 0
        no_price  = market.get('no_price')  or market.get('no_bid')  or 0
        if yes_price is None:
            yes_price = 0
        if no_price is None:
            no_price = 0
        if yes_price == 0 and no_price == 0:
            continue
        if yes_price == 0:
            yes_price = 100 - no_price
        if no_price == 0:
            no_price = 100 - yes_price

        if not (MIN_YES_PRICE <= yes_price <= MAX_YES_PRICE):
            continue

        # Filter: must have minimum volume
        volume = market.get('volume', 0) or 0
        if volume < MIN_VOLUME:
            continue

        # Compute span
        span_hours = _compute_span_hours(
            market.get('open_time', ''),
            market.get('close_time', ''),
        )

        # Determine price for our side
        price_cents = yes_price if side == 'yes' else no_price

        # Build rationale
        dom_vol = a.yes_large_volume if side == 'yes' else a.no_large_volume
        dom_cnt = a.yes_large_count  if side == 'yes' else a.no_large_count
        rationale = (
            f"FTL: {dom_cnt} large trades ({dom_vol} contracts) on {side.upper()} "
            f"in last {lookback_minutes}min | score={score:.1f} | "
            f"span={span_hours:.1f}h | price={price_cents}¢"
        )

        signals.append(LeaderSignal(
            ticker=ticker,
            event_ticker=market.get('event_ticker', ticker),
            side=side.upper(),
            score=score,
            large_trade_count=dom_cnt,
            total_leader_volume=dom_vol,
            price_cents=price_cents,
            market_span_hours=span_hours,
            close_time=market.get('close_time', ''),
            rationale=rationale,
        ))

        if len(signals) >= TOP_N_LEADERS:
            break

    signals.sort(key=lambda s: -s.score)
    logger.info(
        f"FTL: {len(signals)} leader signals generated "
        f"(top score: {signals[0].score:.1f} on {signals[0].ticker})"
        if signals else "FTL: no leader signals after filtering"
    )
    return signals


# ============================================================
# Trade execution
# ============================================================

async def _execute_leader_trade(
    signal: LeaderSignal,
    kalshi_client: KalshiClient,
    db_manager: DatabaseManager,
    bankroll: float,
    paper_mode: bool = True,
) -> bool:
    """
    Execute a Follow the Leader trade.
    Uses a conservative Kelly fraction to size the position.
    Returns True if order was placed successfully.
    """
    try:
        # Check for existing position
        existing = await db_manager.get_position_by_market_and_side(
            signal.ticker, signal.side,
        )
        if existing:
            logger.debug(f"FTL: already hold {signal.ticker} {signal.side} — skip")
            return False

        # Size the position
        price_dollars = signal.price_cents / 100.0
        if price_dollars <= 0:
            return False

        # Kelly: p = implied probability from price, b = (1-p)/p
        p = price_dollars
        b = (1 - p) / p if p < 1 else 0
        # We assume the leader has some edge — model as p_true = p + edge_estimate
        # where edge_estimate is proportional to the score (capped at 10%)
        edge_estimate = min(0.10, signal.score / 100.0)
        p_true = min(0.95, p + edge_estimate)
        kelly = (b * p_true - (1 - p_true)) / b if b > 0 else 0
        kelly = max(0, kelly) * KELLY_FRACTION

        position_dollars = min(
            kelly * bankroll,
            MAX_POSITION_DOLLARS,
        )
        if position_dollars < 1.0:
            logger.debug(f"FTL: position too small (${position_dollars:.2f}) for {signal.ticker}")
            return False

        shares = max(1, int(position_dollars / price_dollars))
        actual_dollars = shares * price_dollars

        # Fee check
        fee = kalshi_taker_fee(signal.price_cents)
        net_edge = edge_estimate - fee
        if net_edge < 0.02:
            logger.debug(f"FTL: net edge {net_edge:.1%} too low after fees for {signal.ticker}")
            return False

        # Limit price: match current market price (taker)
        limit_price = signal.price_cents

        if paper_mode:
            logger.info(
                f"FTL [PAPER]: {signal.ticker} — {shares} {signal.side} @ {limit_price}¢ "
                f"(${actual_dollars:.2f}) | score={signal.score:.1f} | {signal.rationale}"
            )
            return True

        # Live order
        client_order_id = str(uuid.uuid4())
        side_lower = signal.side.lower()
        order_kwargs = {
            "ticker": signal.ticker,
            "client_order_id": client_order_id,
            "side": side_lower,
            "action": "buy",
            "count": shares,
            "type_": "limit",
        }
        if side_lower == "yes":
            order_kwargs["yes_price"] = limit_price
        else:
            order_kwargs["no_price"] = limit_price

        logger.info(
            f"FTL [LIVE]: {signal.ticker} — {shares} {signal.side} @ {limit_price}¢ "
            f"(${actual_dollars:.2f}) | score={signal.score:.1f}"
        )
        order_response = await kalshi_client.place_order(**order_kwargs)

        if order_response and "order" in order_response:
            order_id = order_response["order"].get("order_id", client_order_id)
            position = Position(
                market_id=signal.ticker,
                side=signal.side,
                quantity=shares,
                entry_price=limit_price / 100.0,
                live=True,
                timestamp=datetime.now(),
                rationale=signal.rationale,
                strategy="follow_the_leader",
                stop_loss_price=0.01,
                take_profit_price=0.99,
                max_hold_hours=min(signal.market_span_hours, 72),
            )
            await db_manager.add_position(position)
            logger.info(
                f"FTL ORDER PLACED: {signal.ticker} — Order ID: {order_id}, "
                f"{shares} {signal.side} @ {limit_price}¢"
            )
            return True
        else:
            logger.error(f"FTL order failed: {order_response}")
            return False

    except Exception as e:
        logger.error(f"FTL: error executing trade for {signal.ticker}: {e}")
        return False


# ============================================================
# Main cycle
# ============================================================

async def run_follow_the_leader_cycle(
    kalshi_client: KalshiClient,
    db_manager: DatabaseManager,
    paper_mode: bool = True,
) -> Dict:
    """
    Run one complete Follow the Leader trading cycle.

    1. Fetch and aggregate recent public trades
    2. Score markets by leader signal strength
    3. Filter to low-frequency markets (span ≥ 1 hour)
    4. Execute top signals (up to TOP_N_LEADERS)
    5. Return summary results
    """
    strategy_tag = "follow_the_leader"
    logger.info(
        f"FTL: Starting Follow the Leader cycle "
        f"(paper={paper_mode}, top_n={TOP_N_LEADERS}, "
        f"min_span={MIN_EVENT_SPAN_HOURS}h, lookback={LOOKBACK_MINUTES}min)"
    )

    results: Dict = {
        "markets_scanned": 0,
        "signals_generated": 0,
        "orders_placed": 0,
        "total_position_value": 0.0,
        "paper_mode": paper_mode,
    }

    # Get bankroll
    if paper_mode:
        bankroll = 1000.0
    else:
        try:
            bankroll = await kalshi_client.get_total_portfolio_value()
        except Exception as e:
            logger.error(f"FTL: could not fetch portfolio value: {e}")
            return results
        if bankroll < 10.0:
            logger.warning(f"FTL: insufficient bankroll ${bankroll:.2f}")
            return results

    # Generate signals
    signals = await _generate_leader_signals(kalshi_client, LOOKBACK_MINUTES)
    results["signals_generated"] = len(signals)

    if not signals:
        logger.info("FTL: no leader signals this cycle")
        return results

    # Execute top signals
    orders_placed = 0
    total_value = 0.0

    for signal in signals:
        if orders_placed >= TOP_N_LEADERS:
            break

        placed = await _execute_leader_trade(
            signal, kalshi_client, db_manager, bankroll, paper_mode,
        )
        if placed:
            orders_placed += 1
            total_value += signal.price_cents / 100.0 * signal.large_trade_count

    results["orders_placed"] = orders_placed
    results["total_position_value"] = total_value

    logger.info(
        f"FTL: Cycle complete — {results['signals_generated']} signals, "
        f"{orders_placed} orders placed, ${total_value:.2f} deployed"
    )
    return results
