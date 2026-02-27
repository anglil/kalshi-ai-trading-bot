"""
Follow the Leader Backtest
===========================
Strategy logic:
- Every N minutes (simulated), look at the last window of public trades
- Identify "leaders": traders placing large concentrated bets (high count * price volume)
  on a single side in low-frequency markets (span >= 1 hour)
- Track the top 100 leaders by total volume
- Follow their dominant direction with a fixed $5 position size
- Close position at market settlement, record P&L

Data source:
- Global public trades endpoint (all markets, paginated)
- Settled markets with known results
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import traceback

sys.path.insert(0, '.')
from src.clients.kalshi_client import KalshiClient

# ── Config ──────────────────────────────────────────────────────────────────
POSITION_SIZE = 5.0          # dollars per trade
MIN_SPAN_HOURS = 1.0         # minimum market duration
SIGNAL_WINDOW_MINUTES = 60   # look back 60 min of trades to find leaders
MIN_LEADER_VOLUME = 50       # minimum count*price to qualify as a leader trade
TOP_N_LEADERS = 100          # follow top 100 leader signals
MIN_CONFIDENCE = 0.60        # minimum directional consensus to enter
FEE_RATE = 0.01              # 1% fee estimate

# Series prefixes that actually have public trade data
GOOD_SERIES = ['KXHIGH', 'KXNBA', 'KXCPI', 'KXGDP', 'KXGREENLAND',
               'KXKHAMENEI', 'KXMEDIA', 'KXPGA', 'KXSOCCER', 'KXNHL',
               'KXUCL', 'KXTRUMP', 'KXSENATE', 'KXHOUSE', 'KXFED',
               'KXINFL', 'KXUNEMPLOYMENT', 'KXOIL', 'KXGAS']


async def fetch_global_trades(client, limit=1000, cursor=None):
    """Fetch global public trades across all markets."""
    params = {'limit': limit}
    if cursor:
        params['cursor'] = cursor
    try:
        resp = await client._make_authenticated_request(
            'GET', '/trade-api/v2/trades', params=params
        )
        return resp.get('trades', []), resp.get('cursor')
    except Exception as e:
        return [], None


async def fetch_settled_markets_by_series(client, series_prefix, limit=200):
    """Fetch settled markets for a specific series."""
    try:
        resp = await client._make_authenticated_request(
            'GET', '/trade-api/v2/markets',
            params={'series_ticker': series_prefix, 'limit': limit, 'status': 'settled'}
        )
        return resp.get('markets', [])
    except Exception:
        return []


async def fetch_market_trades(client, ticker, limit=500):
    """Fetch public trades for a single market."""
    try:
        resp = await client._make_authenticated_request(
            'GET', f'/trade-api/v2/markets/{ticker}/trades',
            params={'limit': limit}
        )
        return resp.get('trades', [])
    except Exception:
        return []


def parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace('Z', '+00:00'))
    except:
        return None


def market_span_hours(m):
    t_open = parse_dt(m.get('open_time', ''))
    t_close = parse_dt(m.get('close_time', '') or m.get('expiration_time', ''))
    if t_open and t_close:
        return (t_close - t_open).total_seconds() / 3600
    return 0


def simulate_ftl_on_market(market, trades, position_size=POSITION_SIZE):
    """
    Simulate Follow the Leader on a single settled market.
    
    For each 60-minute window of trades before the market closes,
    identify leader signals and simulate entry + settlement.
    
    Returns list of trade records: {entry_time, side, price, result, pnl, confidence}
    """
    if not trades:
        return []

    result = market.get('result', '')  # 'yes' or 'no'
    if result not in ('yes', 'no'):
        return []

    close_time = parse_dt(market.get('close_time', '') or market.get('expiration_time', ''))
    open_time = parse_dt(market.get('open_time', ''))
    if not close_time or not open_time:
        return []

    span_hours = (close_time - open_time).total_seconds() / 3600
    if span_hours < MIN_SPAN_HOURS:
        return []

    # Sort trades by timestamp ascending
    sorted_trades = sorted(
        [t for t in trades if t.get('created_time')],
        key=lambda t: t['created_time']
    )
    if not sorted_trades:
        return []

    # Simulate: at each hour mark, look at last 60 min of trades
    # and decide whether to enter
    sim_trades = []
    window = timedelta(minutes=SIGNAL_WINDOW_MINUTES)

    # Only consider entry points up to 2 hours before close
    # (need time for the market to resolve)
    entry_cutoff = close_time - timedelta(hours=2)

    # Generate candidate entry times: every hour from open to cutoff
    t = open_time + timedelta(hours=1)
    while t <= entry_cutoff:
        # Gather trades in the window [t - window, t]
        window_start = t - window
        window_trades = [
            tr for tr in sorted_trades
            if window_start <= parse_dt(tr['created_time']) <= t
        ]

        if len(window_trades) >= 5:
            # Compute leader signals
            yes_volume = sum(
                tr.get('count', 0) * tr.get('yes_price', tr.get('price', 0))
                for tr in window_trades
                if tr.get('taker_side', '') == 'yes'
            )
            no_volume = sum(
                tr.get('count', 0) * (100 - tr.get('yes_price', tr.get('price', 50)))
                for tr in window_trades
                if tr.get('taker_side', '') == 'no'
            )
            total_volume = yes_volume + no_volume

            if total_volume >= MIN_LEADER_VOLUME:
                yes_conf = yes_volume / total_volume if total_volume > 0 else 0.5
                no_conf = 1 - yes_conf

                # Determine dominant side
                if yes_conf >= MIN_CONFIDENCE:
                    signal_side = 'yes'
                    confidence = yes_conf
                    # Entry price: approximate from recent trades
                    recent_yes = [tr.get('yes_price', tr.get('price', 50))
                                  for tr in window_trades[-10:]
                                  if tr.get('taker_side') == 'yes']
                    entry_price = (sum(recent_yes) / len(recent_yes) / 100) if recent_yes else 0.5
                elif no_conf >= MIN_CONFIDENCE:
                    signal_side = 'no'
                    confidence = no_conf
                    recent_no = [100 - tr.get('yes_price', tr.get('price', 50))
                                 for tr in window_trades[-10:]
                                 if tr.get('taker_side') == 'no']
                    entry_price = (sum(recent_no) / len(recent_no) / 100) if recent_no else 0.5
                else:
                    t += timedelta(hours=1)
                    continue

                if entry_price <= 0 or entry_price >= 1:
                    t += timedelta(hours=1)
                    continue

                # Simulate outcome
                shares = position_size / entry_price
                fee = position_size * FEE_RATE

                if result == signal_side:
                    # Win: each share pays $1
                    pnl = shares * 1.0 - position_size - fee
                    won = True
                else:
                    # Loss: position expires worthless
                    pnl = -position_size - fee
                    won = False

                sim_trades.append({
                    'ticker': market['ticker'],
                    'series': market.get('series_ticker', ''),
                    'entry_time': t.isoformat(),
                    'side': signal_side,
                    'entry_price': round(entry_price, 3),
                    'confidence': round(confidence, 3),
                    'shares': round(shares, 2),
                    'position_size': position_size,
                    'result': result,
                    'won': won,
                    'pnl': round(pnl, 4),
                    'fee': round(fee, 4),
                    'span_hours': round(span_hours, 1),
                    'window_trades': len(window_trades),
                    'total_volume': round(total_volume, 1),
                })

        t += timedelta(hours=1)

    return sim_trades


async def main():
    client = KalshiClient()
    print("=" * 60)
    print("Follow the Leader — Backtest")
    print("=" * 60)

    # Step 1: Fetch settled markets from known good series
    print("\n[1/3] Fetching settled markets from key series...")
    all_markets = []
    for series in GOOD_SERIES:
        markets = await fetch_settled_markets_by_series(client, series, limit=100)
        if markets:
            print(f"  {series}: {len(markets)} settled markets")
            all_markets.extend(markets)

    # Filter to low-frequency only
    low_freq = [m for m in all_markets if market_span_hours(m) >= MIN_SPAN_HOURS]
    print(f"\nTotal settled low-freq markets: {len(low_freq)}")

    # Step 2: Fetch trades for each market
    print("\n[2/3] Fetching public trades per market...")
    market_trades = {}
    for i, m in enumerate(low_freq):
        ticker = m['ticker']
        trades = await fetch_market_trades(client, ticker, limit=500)
        if trades:
            market_trades[ticker] = trades
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(low_freq)} done, {len(market_trades)} with data")

    print(f"\nMarkets with trade data: {len(market_trades)}")

    await client.close()

    # Step 3: Run simulation
    print("\n[3/3] Running Follow the Leader simulation...")
    all_sim_trades = []
    for m in low_freq:
        ticker = m['ticker']
        trades = market_trades.get(ticker, [])
        sim = simulate_ftl_on_market(m, trades)
        all_sim_trades.extend(sim)

    print(f"\nTotal simulated trades: {len(all_sim_trades)}")

    if not all_sim_trades:
        print("No trades generated — not enough data.")
        return

    # Compute stats
    total_pnl = sum(t['pnl'] for t in all_sim_trades)
    total_invested = sum(t['position_size'] for t in all_sim_trades)
    wins = [t for t in all_sim_trades if t['won']]
    losses = [t for t in all_sim_trades if not t['won']]
    win_rate = len(wins) / len(all_sim_trades) * 100
    avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
    profit_factor = (sum(t['pnl'] for t in wins) /
                     abs(sum(t['pnl'] for t in losses))) if losses else float('inf')

    # Cumulative P&L over time
    sorted_trades = sorted(all_sim_trades, key=lambda t: t['entry_time'])
    cumulative = []
    running = 0
    for t in sorted_trades:
        running += t['pnl']
        cumulative.append({'time': t['entry_time'], 'cum_pnl': round(running, 4)})

    # By series
    by_series = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    for t in all_sim_trades:
        s = t['series']
        by_series[s]['trades'] += 1
        by_series[s]['wins'] += int(t['won'])
        by_series[s]['pnl'] += t['pnl']

    # By confidence bucket
    conf_buckets = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    for t in all_sim_trades:
        bucket = f"{int(t['confidence'] * 10) * 10}%-{int(t['confidence'] * 10) * 10 + 10}%"
        conf_buckets[bucket]['trades'] += 1
        conf_buckets[bucket]['wins'] += int(t['won'])
        conf_buckets[bucket]['pnl'] += t['pnl']

    results = {
        'summary': {
            'total_trades': len(all_sim_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate_pct': round(win_rate, 1),
            'total_pnl': round(total_pnl, 2),
            'total_invested': round(total_invested, 2),
            'roi_pct': round(total_pnl / total_invested * 100, 2) if total_invested else 0,
            'avg_win': round(avg_win, 4),
            'avg_loss': round(avg_loss, 4),
            'profit_factor': round(profit_factor, 3),
            'position_size': POSITION_SIZE,
            'min_confidence': MIN_CONFIDENCE,
            'min_span_hours': MIN_SPAN_HOURS,
        },
        'by_series': {k: dict(v) for k, v in by_series.items()},
        'by_confidence': {k: dict(v) for k, v in conf_buckets.items()},
        'cumulative_pnl': cumulative,
        'trades': sorted_trades[:200],  # first 200 for display
    }

    with open('/tmp/backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("BACKTEST RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total Trades:    {len(all_sim_trades)}")
    print(f"Win Rate:        {win_rate:.1f}%")
    print(f"Total P&L:       ${total_pnl:.2f}")
    print(f"Total Invested:  ${total_invested:.2f}")
    print(f"ROI:             {total_pnl/total_invested*100:.2f}%" if total_invested else "ROI: N/A")
    print(f"Avg Win:         ${avg_win:.4f}")
    print(f"Avg Loss:        ${avg_loss:.4f}")
    print(f"Profit Factor:   {profit_factor:.3f}")
    print(f"\nResults saved to /tmp/backtest_results.json")


if __name__ == '__main__':
    asyncio.run(main())
