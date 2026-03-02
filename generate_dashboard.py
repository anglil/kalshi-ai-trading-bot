"""
Generate a rich interactive trading performance dashboard from live Kalshi data.
Mirrors what the Streamlit dashboard at localhost:8501 would show.
"""

import json
import asyncio
import sys
import os
from datetime import datetime, timezone
from collections import defaultdict

sys.path.insert(0, '.')
from src.clients.kalshi_client import KalshiClient


async def fetch_data():
    client = KalshiClient()
    data = {}
    data['balance'] = await client.get_balance()
    data['positions'] = await client.get_positions()
    # Fetch ALL fills (paginated) — same pattern as settlements
    all_fills = []
    cursor = None
    while True:
        params = {'limit': 1000}
        if cursor:
            params['cursor'] = cursor
        result = await client._make_authenticated_request(
            'GET', '/trade-api/v2/portfolio/fills', params=params
        )
        fills = result.get('fills', [])
        all_fills.extend(fills)
        cursor = result.get('cursor')
        if not cursor or not fills:
            break
    data['fills'] = {'fills': all_fills}  # wrap in dict to match existing code
    data['orders'] = await client.get_orders()

    # Fetch ALL settlements (paginated)
    all_settlements = []
    cursor = None
    while True:
        params = {'limit': 100}
        if cursor:
            params['cursor'] = cursor
        result = await client._make_authenticated_request(
            'GET', '/trade-api/v2/portfolio/settlements', params=params
        )
        settlements = result.get('settlements', [])
        all_settlements.extend(settlements)
        cursor = result.get('cursor')
        if not cursor or not settlements:
            break
    data['settlements'] = all_settlements

    await client.close()
    return data


SNAPSHOT_FILE = os.path.join(os.path.dirname(__file__), 'portfolio_snapshots.json')


def load_snapshots():
    """Load accumulated portfolio snapshots from disk."""
    if os.path.exists(SNAPSHOT_FILE):
        try:
            with open(SNAPSHOT_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_snapshot(balance_usd, portfolio_value_usd):
    """Append a new snapshot with the current API values."""
    snapshots = load_snapshots()
    now_ts = int(datetime.now(tz=timezone.utc).timestamp())
    total = round(balance_usd + portfolio_value_usd, 2)
    # Don't save duplicate if last snapshot is within 5 minutes
    if snapshots and abs(snapshots[-1]['ts'] - now_ts) < 300:
        snapshots[-1] = {'ts': now_ts, 'total': total, 'cash': round(balance_usd, 2), 'positions': round(portfolio_value_usd, 2)}
    else:
        snapshots.append({'ts': now_ts, 'total': total, 'cash': round(balance_usd, 2), 'positions': round(portfolio_value_usd, 2)})
    with open(SNAPSHOT_FILE, 'w') as f:
        json.dump(snapshots, f)
    return snapshots


KNOWN_DEPOSIT = 400.0  # User's actual starting deposit


def build_portfolio_timeline(fills, settlements, balance_now, portfolio_value_now):
    """
    Build the portfolio value curve: cash + position_cost_basis at each event.

    Key design:
      - Deposit is $400 (user-confirmed).
      - Cash is replayed from fills/settlements.
      - Settlement fees are subtracted (separate from fill fees).
      - For settled tickers, use settlement's yes_total_cost/no_total_cost as
        the authoritative cost basis (fixes missing-fill gap).
      - Cost basis is tracked PER TICKER to prevent double-counting.
      - Final point is anchored to the live API value.
    """
    from dateutil.parser import isoparse
    deposit = KNOWN_DEPOSIT
    total_now = balance_now + portfolio_value_now

    # --- Step 0: Build authoritative cost basis from settlements ---
    # Settlements report the TRUE cost Kalshi tracked, which includes fills
    # the API may not have returned to us.
    settlement_cost_basis = {}  # ticker -> total cost in USD
    for s in settlements:
        ticker = s.get('ticker', '')
        yes_cost = s.get('yes_total_cost', 0) / 100.0  # cents -> USD
        no_cost = s.get('no_total_cost', 0) / 100.0
        settlement_cost_basis[ticker] = yes_cost + no_cost

    # --- Step 1: Parse all events ---
    all_events = []

    for f in fills:
        ts = f.get('ts', 0)
        if ts == 0:
            continue
        count = f.get('count', 0)
        side = f.get('side', 'yes')
        if side == 'yes':
            price_cents = f.get('yes_price', 0) or f.get('price', 0) or 0
        else:
            price_cents = f.get('no_price', 0) or f.get('price', 0) or 0
        cost = (count * price_cents) / 100.0
        fee_raw = float(f.get('fee_cost', 0) or f.get('taker_fee', 0) or 0)
        fee = fee_raw / 100.0 if fee_raw > 1.0 else fee_raw
        action = f.get('action', 'buy')
        ticker = f.get('ticker', '')
        all_events.append({
            'ts': ts, 'type': 'fill', 'action': action,
            'cost': cost, 'fee': fee, 'ticker': ticker,
        })

    for s in settlements:
        settled_str = s.get('settled_time', '')
        try:
            settled_dt = isoparse(settled_str)
            ts = int(settled_dt.timestamp())
        except Exception:
            continue
        revenue = s.get('revenue', 0) / 100.0
        ticker = s.get('ticker', '')
        # Settlement fee — separate from fill fees
        sett_fee = float(s.get('fee_cost', 0) or 0)
        all_events.append({
            'ts': ts, 'type': 'settlement',
            'revenue': revenue, 'ticker': ticker,
            'settlement_fee': sett_fee,
        })

    if not all_events:
        return [{'ts': 0, 'label': 'Now', 'total': total_now}], deposit

    # Sort: settlements before fills at the same timestamp
    all_events.sort(key=lambda x: (x['ts'], 0 if x['type'] == 'settlement' else 1))

    # --- Step 2: Replay cash, track cost basis PER TICKER ---
    cash = deposit
    ticker_cost = defaultdict(float)  # cost basis per ticker
    # Track BUY-ONLY cost per ticker (not net of sells) for missing-fill detection.
    # The fills API is missing some buy fills. We detect this by comparing
    # our buy total against the settlement's authoritative cost.
    # Sells are tracked separately and already returned cash to the replay.
    fill_buy_cost_per_ticker = defaultdict(float)

    t_start = all_events[0]['ts']
    t_end = int(datetime.now(tz=timezone.utc).timestamp())

    raw_points = []
    raw_points.append({'ts': t_start - 3600, 'cash': deposit, 'positions': 0.0})

    for ev in all_events:
        if ev['type'] == 'fill':
            cash -= ev['fee']
            ticker = ev.get('ticker', '')
            if ev['action'] == 'buy':
                cash -= ev['cost']
                ticker_cost[ticker] += ev['cost']
                fill_buy_cost_per_ticker[ticker] += ev['cost']
            else:  # sell
                cash += ev['cost']
                ticker_cost[ticker] = max(0, ticker_cost[ticker] - ev['cost'])
        elif ev['type'] == 'settlement':
            ticker = ev.get('ticker', '')
            # Subtract settlement fee from cash
            cash -= ev.get('settlement_fee', 0)

            # Use settlement's authoritative cost basis to fix missing fills.
            # Compare settlement cost against BUY-ONLY fills (not net of sells).
            # Sells already returned cash to the replay, so they don't affect
            # the missing-buy calculation.
            auth_cost = settlement_cost_basis.get(ticker, 0)
            our_buy_cost = fill_buy_cost_per_ticker.get(ticker, 0)
            missing_fill_cost = max(0, auth_cost - our_buy_cost)
            if missing_fill_cost > 0:
                cash -= missing_fill_cost
                # Also add the missing cost to ticker_cost so positions value
                # reflects the true cost basis before this settlement zeroes it
                ticker_cost[ticker] += missing_fill_cost

            # Add settlement revenue
            cash += ev['revenue']
            # Zero out this ticker's cost basis — position is closed
            ticker_cost[ticker] = 0
            fill_buy_cost_per_ticker[ticker] = 0

        total_positions = sum(ticker_cost.values())
        raw_points.append({
            'ts': ev['ts'],
            'cash': cash,
            'positions': total_positions,
        })

    # --- Step 3: Build timeline ---
    # The replayed cash should now closely match the API cash because we've
    # accounted for settlement fees and missing fills. Any remaining small
    # gap is from rounding. No proportional correction needed.
    timeline_points = []
    for rp in raw_points:
        total = rp['cash'] + rp['positions']
        timeline_points.append({'ts': rp['ts'], 'total': round(total, 2)})

    # Anchor the final point to the live API value
    timeline_points.append({'ts': t_end, 'total': round(total_now, 2)})

    # --- Step 4: Deduplicate timestamps (keep last value per timestamp) ---
    seen = {}
    for rp in timeline_points:
        seen[rp['ts']] = rp
    timeline_points = sorted(seen.values(), key=lambda x: x['ts'])

    # Thin to max 500 points if needed
    MAX_POINTS = 500
    if len(timeline_points) > MAX_POINTS:
        step = len(timeline_points) / MAX_POINTS
        thinned = [timeline_points[0]]
        for i in range(1, MAX_POINTS - 1):
            idx = int(i * step)
            if idx < len(timeline_points):
                thinned.append(timeline_points[idx])
        thinned.append(timeline_points[-1])
        timeline_points = thinned

    # --- Step 5: Format output ---
    timeline = []
    for rp in timeline_points:
        dt = datetime.fromtimestamp(rp['ts'], tz=timezone.utc)
        timeline.append({
            'ts': rp['ts'],
            'label': dt.strftime('%b %d %H:%M'),
            'total': rp['total'],
        })

    if timeline:
        timeline[-1]['total'] = round(total_now, 2)
        timeline[-1]['label'] = datetime.now(tz=timezone.utc).strftime('%b %d %H:%M')

    return timeline, deposit


def kalshi_market_url(ticker):
    """
    Build the Kalshi market page URL from a ticker.
    URL pattern: https://kalshi.com/markets/{series_lower}/{series_slug}/{ticker_lower}
    The series ticker is the alphabetic prefix of the market ticker (before the first digit).
    """
    import re
    ticker = ticker.strip()
    # Extract series ticker: everything up to the first digit
    match = re.match(r'^([A-Za-z]+)', ticker)
    series = match.group(1).lower() if match else ticker.lower()
    # Build a human-readable slug from the series (strip leading 'kx')
    slug_map = {
        'kxhighchi': 'highest-temperature-in-chicago',
        'kxhighny': 'highest-temperature-in-new-york',
        'kxhighmia': 'highest-temperature-in-miami',
        'kxhighaus': 'highest-temperature-in-austin',
        'kxhighdal': 'highest-temperature-in-dallas',
        'kxhighla': 'highest-temperature-in-los-angeles',
        'kxhighsea': 'highest-temperature-in-seattle',
        'kxnbagame': 'nba-game-winner',
        'kxcpi': 'cpi-month-over-month',
        'kxgreenland': 'greenland',
        'kxkhameneiout': 'khamenei-out',
        'kxmediareleasest': 'media-release',
    }
    slug = slug_map.get(series, series.replace('kx', '', 1))
    return f"https://kalshi.com/markets/{series}/{slug}/{ticker.lower()}"


def categorize_ticker(ticker):
    """Categorize a market ticker into a strategy/category."""
    t = ticker.upper()
    if 'NBAGAME' in t or 'NBA' in t:
        return 'NBA'
    elif 'SOCCER' in t or 'MLS' in t or 'EPL' in t:
        return 'Soccer'
    elif 'HIGH' in t or 'LOW' in t or 'TEMP' in t or 'PRECIP' in t or 'SNOW' in t:
        return 'Weather'
    elif 'CPI' in t or 'GDP' in t or 'JOBS' in t or 'FED' in t or 'FOMC' in t or 'ECON' in t:
        return 'Economics'
    elif 'GAS' in t or 'OIL' in t or 'EIA' in t:
        return 'Energy'
    elif 'FLU' in t or 'COVID' in t or 'HEALTH' in t:
        return 'Health'
    elif 'GREENLAND' in t or 'ELECTION' in t or 'PRES' in t or 'CONGRESS' in t:
        return 'Politics'
    else:
        return 'Other'


def process_data(data):
    """Process raw Kalshi API data into dashboard metrics."""
    balance_cents = data['balance'].get('balance', 0)
    portfolio_value_cents = data['balance'].get('portfolio_value', 0)
    balance_usd = balance_cents / 100
    portfolio_value_usd = portfolio_value_cents / 100
    total_value_usd = balance_usd + portfolio_value_usd

    # Build portfolio timeline (using fills AND settlements for accurate reconstruction)
    all_fills = data['fills'].get('fills', [])
    all_settlements = data.get('settlements', [])
    portfolio_timeline, deposit = build_portfolio_timeline(all_fills, all_settlements, balance_usd, portfolio_value_usd)

    # Save a snapshot of the current portfolio value for future accumulation
    save_snapshot(balance_usd, portfolio_value_usd)

    # Process event positions
    event_positions = data['positions'].get('event_positions', [])
    market_positions = data['positions'].get('market_positions', [])

    # Aggregate metrics
    total_realized_pnl = sum(p.get('realized_pnl', 0) for p in event_positions) / 100
    total_fees = sum(p.get('fees_paid', 0) for p in event_positions) / 100
    total_exposure = sum(p.get('event_exposure', 0) for p in event_positions) / 100
    total_cost = sum(p.get('total_cost', 0) for p in event_positions) / 100

    # Open positions — use market_positions for accurate count (event_positions groups by event)
    open_market_positions = [p for p in market_positions if p.get('position', 0) != 0]
    # Keep event-level for P&L tracking
    open_positions = [p for p in event_positions if p.get('event_exposure', 0) > 0]
    closed_positions = [p for p in event_positions if p.get('event_exposure', 0) == 0]

    # Category breakdown
    category_pnl = defaultdict(float)
    category_exposure = defaultdict(float)
    category_cost = defaultdict(float)
    for p in event_positions:
        cat = categorize_ticker(p['event_ticker'])
        category_pnl[cat] += p.get('realized_pnl', 0) / 100
        category_exposure[cat] += p.get('event_exposure', 0) / 100
        category_cost[cat] += p.get('total_cost', 0) / 100

    # Process fills for trade history
    fills = data['fills'].get('fills', [])
    fills_sorted = sorted(fills, key=lambda x: x.get('ts', 0))

    # Build trade timeline (cumulative cost over time)
    timeline = []
    cumulative_cost = 0
    for fill in fills_sorted:
        ts = fill.get('ts', 0)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
        fill_side = fill.get('side', 'yes')
        if fill_side == 'yes':
            fill_price = fill.get('yes_price', 0) or fill.get('price', 0) or 0
        else:
            fill_price = fill.get('no_price', 0) or fill.get('price', 0) or 0
        cost = fill.get('count', 0) * fill_price
        if fill.get('action') == 'buy':
            cumulative_cost += cost
        else:
            cumulative_cost -= cost
        timeline.append({
            'time': dt,
            'ts': ts,
            'action': fill.get('action', ''),
            'ticker': fill.get('ticker', ''),
            'side': fill_side,
            'count': fill.get('count', 0),
            'price': fill_price,
            'fee': float(fill.get('fee_cost', 0) or fill.get('taker_fee', 0) or 0),
            'category': categorize_ticker(fill.get('ticker', '')),
        })

    # Fill category breakdown
    fill_categories = defaultdict(lambda: {'buys': 0, 'sells': 0, 'volume': 0, 'fees': 0})
    for f in fills:
        cat = categorize_ticker(f.get('ticker', ''))
        fill_categories[cat]['volume'] += f.get('count', 0)
        fill_categories[cat]['fees'] += float(f.get('fee_cost', 0))
        if f.get('action') == 'buy':
            fill_categories[cat]['buys'] += 1
        else:
            fill_categories[cat]['sells'] += 1

    # Win/loss for closed positions
    wins = [p for p in event_positions if p.get('realized_pnl', 0) > 0]
    losses = [p for p in event_positions if p.get('realized_pnl', 0) < 0]
    neutral = [p for p in event_positions if p.get('realized_pnl', 0) == 0]

    win_rate = len(wins) / len(event_positions) * 100 if event_positions else 0

    # Hourly activity from fills
    hourly = defaultdict(int)
    for f in fills:
        ts = f.get('ts', 0)
        hour = datetime.fromtimestamp(ts, tz=timezone.utc).hour
        hourly[hour] += 1

    return {
        'balance_usd': balance_usd,
        'portfolio_value_usd': portfolio_value_usd,
        'total_value_usd': total_value_usd,
        'total_realized_pnl': total_realized_pnl,
        'total_fees': total_fees,
        'total_exposure': total_exposure,
        'total_cost': total_cost,
        'open_positions': open_positions,
        'open_market_positions': open_market_positions,
        'closed_positions': closed_positions,
        'event_positions': event_positions,
        'market_positions': market_positions,
        'category_pnl': dict(category_pnl),
        'category_exposure': dict(category_exposure),
        'category_cost': dict(category_cost),
        'fills': fills,
        'timeline': timeline,
        'fill_categories': dict(fill_categories),
        'wins': wins,
        'losses': losses,
        'neutral': neutral,
        'win_rate': win_rate,
        'total_trades': len(fills),
        'hourly': dict(hourly),
        'orders': data['orders'].get('orders', []),
        'portfolio_timeline': portfolio_timeline,
        'deposit': deposit,
    }


def generate_html(metrics, generated_at):
    """Generate the full interactive HTML dashboard."""

    # Prepare chart data as JSON
    # 1. Category P&L bar chart
    cat_pnl = metrics['category_pnl']
    cat_labels = list(cat_pnl.keys())
    cat_values = [round(v, 2) for v in cat_pnl.values()]
    cat_colors = ['#22c55e' if v >= 0 else '#ef4444' for v in cat_values]

    # 2. Category exposure donut
    cat_exp = metrics['category_exposure']
    exp_labels = [k for k, v in cat_exp.items() if v > 0]
    exp_values = [round(v, 2) for k, v in cat_exp.items() if v > 0]

    # 3. Trade timeline (scatter by category)
    timeline = metrics['timeline']
    tl_times = [t['time'] for t in timeline]
    tl_prices = [round(t['price'] * 100) for t in timeline]
    tl_actions = [t['action'] for t in timeline]
    tl_tickers = [t['ticker'] for t in timeline]
    tl_cats = [t['category'] for t in timeline]

    # 4. Hourly activity heatmap
    hourly = metrics['hourly']
    hours = list(range(24))
    hour_counts = [hourly.get(h, 0) for h in hours]

    # 5. Win/Loss/Neutral pie
    wl_labels = ['Profitable', 'Loss', 'Neutral/Open']
    wl_values = [len(metrics['wins']), len(metrics['losses']), len(metrics['neutral'])]
    wl_colors = ['#22c55e', '#ef4444', '#94a3b8']

    # 6. Open positions table data
    open_pos_rows = []
    for p in sorted(metrics['open_positions'], key=lambda x: -x.get('event_exposure', 0)):
        cat = categorize_ticker(p['event_ticker'])
        pnl = p.get('realized_pnl', 0) / 100
        pnl_class = 'text-green-400' if pnl > 0 else ('text-red-400' if pnl < 0 else 'text-slate-400')
        open_pos_rows.append({
            'ticker': p['event_ticker'],
            'kalshi_url': kalshi_market_url(p['event_ticker']),
            'category': cat,
            'exposure': p.get('event_exposure_dollars', '0.00'),
            'cost': p.get('total_cost_dollars', '0.00'),
            'shares': p.get('total_cost_shares', 0),
            'realized_pnl': f"${pnl:+.2f}",
            'fees': p.get('fees_paid_dollars', '0.00'),
            'pnl_class': pnl_class,
        })

    # 7. Recent fills table
    recent_fills = sorted(metrics['fills'], key=lambda x: -x.get('ts', 0))[:20]
    fill_rows = []
    for f in recent_fills:
        ts = f.get('ts', 0)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%m/%d %H:%M')
        action_class = 'text-green-400' if f.get('action') == 'buy' else 'text-red-400'
        fill_ticker = f.get('ticker', '')
        fill_rows.append({
            'time': dt,
            'ticker': fill_ticker,
            'kalshi_url': kalshi_market_url(fill_ticker) if fill_ticker else '#',
            'action': f.get('action', '').upper(),
            'side': f.get('side', '').upper(),
            'count': f.get('count', 0),
            'price': f"${f.get('price', 0)*100:.0f}¢",
            'fee': f"${float(f.get('fee_cost', 0)):.2f}",
            'action_class': action_class,
            'category': categorize_ticker(fill_ticker),
        })

    # 8. Category volume bar
    fc = metrics['fill_categories']
    fc_labels = list(fc.keys())
    fc_buys = [fc[k]['buys'] for k in fc_labels]
    fc_sells = [fc[k]['sells'] for k in fc_labels]

    # 9. Portfolio value over time
    ptl = metrics['portfolio_timeline']
    ptl_labels = [p['label'] for p in ptl]
    ptl_total = [p['total'] for p in ptl]
    deposit = round(metrics['deposit'], 2)

    # Summary stats
    net_pnl = metrics['total_realized_pnl']
    net_pnl_class = 'text-green-400' if net_pnl >= 0 else 'text-red-400'
    net_pnl_str = f"${net_pnl:+.2f}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kalshi AI Trading Bot — Live Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body {{ background: #0f172a; color: #e2e8f0; font-family: 'Inter', system-ui, sans-serif; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; }}
  .metric-card {{ background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border: 1px solid #334155; border-radius: 12px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600; }}
  .badge-green {{ background: #166534; color: #86efac; }}
  .badge-red {{ background: #7f1d1d; color: #fca5a5; }}
  .badge-blue {{ background: #1e3a5f; color: #93c5fd; }}
  .badge-yellow {{ background: #713f12; color: #fde68a; }}
  .badge-purple {{ background: #4c1d95; color: #c4b5fd; }}
  .badge-gray {{ background: #1e293b; color: #94a3b8; }}
  .live-dot {{ width: 8px; height: 8px; background: #22c55e; border-radius: 50%; display: inline-block; animation: pulse 2s infinite; }}
  @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }} }}
  .scrollable {{ max-height: 320px; overflow-y: auto; }}
  .scrollable::-webkit-scrollbar {{ width: 4px; }}
  .scrollable::-webkit-scrollbar-track {{ background: #0f172a; }}
  .scrollable::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 2px; }}
  canvas {{ max-height: 280px; }}
</style>
</head>
<body class="min-h-screen p-4">

<!-- Header -->
<div class="flex items-center justify-between mb-6">
  <div class="flex items-center gap-3">
    <div class="text-2xl">🚀</div>
    <div>
      <h1 class="text-xl font-bold text-white">Kalshi AI Trading Bot</h1>
      <p class="text-xs text-slate-400">Beast Mode Dashboard — Live Performance</p>
    </div>
  </div>
  <div class="flex items-center gap-2 text-xs text-slate-400">
    <span class="live-dot"></span>
    <span class="text-green-400 font-semibold">LIVE TRADING</span>
    <span class="ml-2">Updated: {generated_at}</span>
  </div>
</div>

<!-- KPI Row -->
<div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
  <div class="metric-card p-4">
    <div class="text-xs text-slate-400 mb-1">Available Cash</div>
    <div class="text-2xl font-bold text-white">${metrics['balance_usd']:.2f}</div>
    <div class="text-xs text-slate-500 mt-1">Uninvested balance</div>
  </div>
  <div class="metric-card p-4">
    <div class="text-xs text-slate-400 mb-1">Portfolio Value</div>
    <div class="text-2xl font-bold text-blue-400">${metrics['portfolio_value_usd']:.2f}</div>
    <div class="text-xs text-slate-500 mt-1">Open position value</div>
  </div>
  <div class="metric-card p-4">
    <div class="text-xs text-slate-400 mb-1">Total Account</div>
    <div class="text-2xl font-bold text-white">${metrics['total_value_usd']:.2f}</div>
    <div class="text-xs text-slate-500 mt-1">Cash + positions</div>
  </div>
  <div class="metric-card p-4">
    <div class="text-xs text-slate-400 mb-1">Realized P&L</div>
    <div class="text-2xl font-bold {net_pnl_class}">{net_pnl_str}</div>
    <div class="text-xs text-slate-500 mt-1">All closed trades</div>
  </div>
</div>

<!-- Second KPI Row -->
<div class="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
  <div class="card p-3 text-center">
    <div class="text-xs text-slate-400">Open Positions</div>
     <div class="text-xl font-bold text-white">{len(metrics['open_market_positions'])}</div>
  </div>
  <div class="card p-3 text-center">
    <div class="text-xs text-slate-400">Total Trades</div>
    <div class="text-xl font-bold text-white">{metrics['total_trades']}</div>
  </div>
  <div class="card p-3 text-center">
    <div class="text-xs text-slate-400">Win Rate</div>
    <div class="text-xl font-bold {'text-green-400' if metrics['win_rate'] >= 50 else 'text-red-400'}">{metrics['win_rate']:.0f}%</div>
  </div>
  <div class="card p-3 text-center">
    <div class="text-xs text-slate-400">Total Exposure</div>
    <div class="text-xl font-bold text-blue-400">${metrics['total_exposure']:.2f}</div>
  </div>
  <div class="card p-3 text-center">
    <div class="text-xs text-slate-400">Fees Paid</div>
    <div class="text-xl font-bold text-yellow-400">${metrics['total_fees']:.2f}</div>
  </div>
</div>

<!-- Portfolio Value Over Time -->
<div class="card p-4 mb-4">
  <div class="flex items-center justify-between mb-1">
    <div class="text-sm font-semibold text-white">📈 Portfolio Value Over Time</div>
    <div class="flex items-center gap-4 text-xs text-slate-400">
      <span><span style="display:inline-block;width:18px;height:3px;background:#ef4444;vertical-align:middle;"></span> Total Portfolio (cash + unrealized)</span>
      <span><span style="display:inline-block;width:18px;height:2px;background:#94a3b8;border-top:2px dashed #94a3b8;vertical-align:middle;"></span> Deposited (${deposit:.0f})</span>
    </div>
  </div>
  <div class="text-xs text-slate-500 mb-3">Cash + Position Value · {ptl_labels[0] if ptl_labels else 'N/A'} → now · {len(ptl)} data points</div>
  <canvas id="portfolioChart" style="max-height:320px;"></canvas>
</div>

<!-- Charts Row 1 -->
<div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">

  <!-- P&L by Category -->
  <div class="card p-4">
    <div class="text-sm font-semibold text-white mb-3">📊 Realized P&L by Strategy</div>
    <canvas id="pnlChart"></canvas>
  </div>

  <!-- Win/Loss Pie -->
  <div class="card p-4">
    <div class="text-sm font-semibold text-white mb-3">🎯 Position Outcomes</div>
    <canvas id="winLossChart"></canvas>
    <div class="flex justify-center gap-4 mt-2 text-xs">
      <span class="text-green-400">✓ {len(metrics['wins'])} Profitable</span>
      <span class="text-red-400">✗ {len(metrics['losses'])} Loss</span>
      <span class="text-slate-400">◎ {len(metrics['neutral'])} Open/Neutral</span>
    </div>
  </div>

  <!-- Exposure Donut -->
  <div class="card p-4">
    <div class="text-sm font-semibold text-white mb-3">💰 Exposure by Strategy</div>
    <canvas id="exposureChart"></canvas>
  </div>

</div>

<!-- Charts Row 2 -->
<div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">

  <!-- Trade Volume by Category -->
  <div class="card p-4">
    <div class="text-sm font-semibold text-white mb-3">📈 Trade Activity by Strategy</div>
    <canvas id="volumeChart"></canvas>
  </div>

  <!-- Hourly Activity -->
  <div class="card p-4">
    <div class="text-sm font-semibold text-white mb-3">🕐 Hourly Trading Activity (UTC)</div>
    <canvas id="hourlyChart"></canvas>
  </div>

</div>

<!-- Open Positions Table -->
<div class="card p-4 mb-4">
  <div class="flex items-center justify-between mb-3">
    <div class="text-sm font-semibold text-white">📋 Open Positions ({len(metrics['open_market_positions'])})</div>
    <span class="badge badge-blue">Live</span>
  </div>
  <div class="scrollable">
    <table class="w-full text-xs">
      <thead>
        <tr class="text-slate-400 border-b border-slate-700">
          <th class="text-left py-2 pr-3">Market</th>
          <th class="text-left py-2 pr-3">Strategy</th>
          <th class="text-right py-2 pr-3">Exposure</th>
          <th class="text-right py-2 pr-3">Cost Basis</th>
          <th class="text-right py-2 pr-3">Shares</th>
          <th class="text-right py-2 pr-3">Realized P&L</th>
          <th class="text-right py-2">Fees</th>
        </tr>
      </thead>
      <tbody>
"""

    for row in open_pos_rows:
        cat_badge = {
            'NBA': 'badge-blue', 'Weather': 'badge-green', 'Economics': 'badge-yellow',
            'Energy': 'badge-purple', 'Politics': 'badge-red', 'Soccer': 'badge-blue',
            'Health': 'badge-green', 'Other': 'badge-gray'
        }.get(row['category'], 'badge-gray')
        html += f"""        <tr class="border-b border-slate-800 hover:bg-slate-800/30">
          <td class="py-2 pr-3 font-mono">
            <a href="{row['kalshi_url']}" target="_blank" rel="noopener"
               class="text-blue-400 hover:text-blue-300 hover:underline" title="Open on Kalshi">{row['ticker']}</a>
          </td>
          <td class="py-2 pr-3"><span class="badge {cat_badge}">{row['category']}</span></td>
          <td class="py-2 pr-3 text-right text-white">${row['exposure']}</td>
          <td class="py-2 pr-3 text-right text-slate-300">${row['cost']}</td>
          <td class="py-2 pr-3 text-right text-slate-300">{row['shares']}</td>
          <td class="py-2 pr-3 text-right {row['pnl_class']}">{row['realized_pnl']}</td>
          <td class="py-2 text-right text-yellow-400">${row['fees']}</td>
        </tr>
"""

    html += """      </tbody>
    </table>
  </div>
</div>

<!-- Recent Fills Table -->
<div class="card p-4 mb-4">
  <div class="flex items-center justify-between mb-3">
    <div class="text-sm font-semibold text-white">🔄 Recent Trade Fills (Last 20)</div>
    <span class="badge badge-green">Live Feed</span>
  </div>
  <div class="scrollable">
    <table class="w-full text-xs">
      <thead>
        <tr class="text-slate-400 border-b border-slate-700">
          <th class="text-left py-2 pr-3">Time (UTC)</th>
          <th class="text-left py-2 pr-3">Market</th>
          <th class="text-left py-2 pr-3">Strategy</th>
          <th class="text-center py-2 pr-3">Action</th>
          <th class="text-center py-2 pr-3">Side</th>
          <th class="text-right py-2 pr-3">Qty</th>
          <th class="text-right py-2 pr-3">Price</th>
          <th class="text-right py-2">Fee</th>
        </tr>
      </thead>
      <tbody>
"""

    for row in fill_rows:
        cat_badge = {
            'NBA': 'badge-blue', 'Weather': 'badge-green', 'Economics': 'badge-yellow',
            'Energy': 'badge-purple', 'Politics': 'badge-red', 'Soccer': 'badge-blue',
            'Health': 'badge-green', 'Other': 'badge-gray'
        }.get(row['category'], 'badge-gray')
        html += f"""        <tr class="border-b border-slate-800 hover:bg-slate-800/30">
          <td class="py-2 pr-3 text-slate-400">{row['time']}</td>
          <td class="py-2 pr-3 font-mono text-xs">
            <a href="{row['kalshi_url']}" target="_blank" rel="noopener"
               class="text-blue-400 hover:text-blue-300 hover:underline" title="Open on Kalshi">{row['ticker'][:30]}</a>
          </td>
          <td class="py-2 pr-3"><span class="badge {cat_badge}">{row['category']}</span></td>
          <td class="py-2 pr-3 text-center"><span class="{row['action_class']} font-bold">{row['action']}</span></td>
          <td class="py-2 pr-3 text-center text-slate-300">{row['side']}</td>
          <td class="py-2 pr-3 text-right text-white">{row['count']}</td>
          <td class="py-2 pr-3 text-right text-slate-300">{row['price']}</td>
          <td class="py-2 text-right text-yellow-400">{row['fee']}</td>
        </tr>
"""

    html += f"""      </tbody>
    </table>
  </div>
</div>

<!-- Bot Status Footer -->
<div class="card p-4 mb-4">
  <div class="text-sm font-semibold text-white mb-3">⚙️ Bot Status</div>
  <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
    <div class="flex items-center gap-2">
      <span class="live-dot"></span>
      <span class="text-green-400 font-semibold">RUNNING</span>
      <span class="text-slate-400">beast_mode_bot.py --live</span>
    </div>
    <div class="text-slate-400">🌙 Night Mode: <span class="text-yellow-400">Active (11PM–7AM ET)</span></div>
    <div class="text-slate-400">🤖 AI Engine: <span class="text-blue-400">Google Gemini</span></div>
    <div class="text-slate-400">🔄 Scan Interval: <span class="text-white">30s</span></div>
    <div class="text-slate-400">📊 Max Positions: <span class="text-white">20</span></div>
    <div class="text-slate-400">🛡️ Max Daily Loss: <span class="text-red-400">15%</span></div>
    <div class="text-slate-400">💸 Daily AI Budget: <span class="text-white">$50</span></div>
    <div class="text-slate-400">⚡ Kelly Fraction: <span class="text-white">0.75x</span></div>
  </div>
</div>

<div class="text-center text-xs text-slate-600 pb-4">
  Kalshi AI Trading Bot Dashboard · Generated {generated_at} · Data from Kalshi REST API
</div>

<script>
const chartDefaults = {{
  color: '#94a3b8',
  plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }} }},
  scales: {{
    x: {{ ticks: {{ color: '#64748b', font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }},
    y: {{ ticks: {{ color: '#64748b', font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }}
  }}
}};

// 1. P&L by Category
new Chart(document.getElementById('pnlChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(cat_labels)},
    datasets: [{{
      label: 'Realized P&L ($)',
      data: {json.dumps(cat_values)},
      backgroundColor: {json.dumps(cat_colors)},
      borderRadius: 6,
    }}]
  }},
  options: {{
    ...chartDefaults,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#64748b', font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ color: '#64748b', font: {{ size: 10 }}, callback: v => '$' + v.toFixed(2) }}, grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});

// 2. Win/Loss Pie
new Chart(document.getElementById('winLossChart'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(wl_labels)},
    datasets: [{{
      data: {json.dumps(wl_values)},
      backgroundColor: {json.dumps(wl_colors)},
      borderWidth: 0,
      hoverOffset: 6
    }}]
  }},
  options: {{
    plugins: {{
      legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 10 }} }} }}
    }},
    cutout: '65%'
  }}
}});

// 3. Exposure Donut
const expColors = ['#3b82f6','#22c55e','#f59e0b','#a855f7','#ef4444','#06b6d4','#f97316','#64748b'];
new Chart(document.getElementById('exposureChart'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(exp_labels)},
    datasets: [{{
      data: {json.dumps(exp_values)},
      backgroundColor: expColors.slice(0, {len(exp_labels)}),
      borderWidth: 0,
      hoverOffset: 6
    }}]
  }},
  options: {{
    plugins: {{
      legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 10 }} }} }}
    }},
    cutout: '60%'
  }}
}});

// 4. Volume by Category
new Chart(document.getElementById('volumeChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(fc_labels)},
    datasets: [
      {{ label: 'Buys', data: {json.dumps(fc_buys)}, backgroundColor: '#22c55e', borderRadius: 4 }},
      {{ label: 'Sells', data: {json.dumps(fc_sells)}, backgroundColor: '#ef4444', borderRadius: 4 }}
    ]
  }},
  options: {{
    ...chartDefaults,
    plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 10 }} }} }} }},
    scales: {{
      x: {{ stacked: false, ticks: {{ color: '#64748b', font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }},
      y: {{ stacked: false, ticks: {{ color: '#64748b', font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});

// 5. Hourly Activity
new Chart(document.getElementById('hourlyChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps([f"{h:02d}:00" for h in hours])},
    datasets: [{{
      label: 'Trades',
      data: {json.dumps(hour_counts)},
      backgroundColor: '#3b82f6',
      borderRadius: 3,
    }}]
  }},
  options: {{
    ...chartDefaults,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#64748b', font: {{ size: 9 }}, maxRotation: 45 }}, grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ color: '#64748b', font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});

// 6. Portfolio Value Over Time — two lines only: total portfolio and deposit baseline
const portfolioLabels = {json.dumps(ptl_labels)};
const portfolioTotal = {json.dumps(ptl_total)};
const depositLine = new Array(portfolioLabels.length).fill({deposit});

new Chart(document.getElementById('portfolioChart'), {{
  type: 'line',
  data: {{
    labels: portfolioLabels,
    datasets: [
      {{
        label: 'Total Portfolio',
        data: portfolioTotal,
        fill: false,
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239,68,68,0.08)',
        borderWidth: 2.5,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.3,
      }},
      {{
        label: 'Deposited (${deposit:.0f})',
        data: depositLine,
        fill: false,
        borderColor: '#94a3b8',
        borderWidth: 1.5,
        borderDash: [6, 4],
        pointRadius: 0,
        tension: 0,
      }}
    ]
  }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }}, usePointStyle: true, boxWidth: 20 }} }},
      tooltip: {{
        backgroundColor: '#1e293b',
        titleColor: '#e2e8f0',
        bodyColor: '#94a3b8',
        borderColor: '#334155',
        borderWidth: 1,
        callbacks: {{
          label: ctx => ` ${{ctx.dataset.label}}: $${{ctx.parsed.y.toFixed(2)}}`,
          afterBody: (items) => {{
            const total = items.find(i => i.dataset.label === 'Total Portfolio');
            if (total) {{
              const pnl = total.parsed.y - {deposit};
              return [`  P&L vs deposit: ${{pnl >= 0 ? '+' : ''}}$${{pnl.toFixed(2)}}`];
            }}
            return [];
          }}
        }}
      }}
    }},
    scales: {{
      x: {{
        ticks: {{ color: '#64748b', font: {{ size: 10 }}, maxTicksLimit: 12, maxRotation: 30 }},
        grid: {{ color: '#1e293b' }}
      }},
      y: {{
        ticks: {{ color: '#64748b', font: {{ size: 10 }}, callback: v => '$' + v.toFixed(0) }},
        grid: {{ color: '#1e293b' }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

    return html


async def main():
    print(f"Python: {sys.version}")
    print(f"CWD: {os.getcwd()}")
    print(f"KALSHI_API_KEY set: {bool(os.getenv('KALSHI_API_KEY'))}")
    print(f"Private key file exists: {os.path.exists('kalshi_private_key')}")
    if os.path.exists('kalshi_private_key'):
        with open('kalshi_private_key', 'r') as f:
            content = f.read()
        print(f"Private key file size: {len(content)} bytes")
        print(f"Private key starts with: {content[:30]}")
        print(f"Private key ends with: ...{content[-30:].strip()}")
    print(f".env file exists: {os.path.exists('.env')}")
    print("Fetching live data from Kalshi API...")
    data = await fetch_data()
    print("Processing metrics...")
    metrics = process_data(data)
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    print("Generating dashboard HTML...")
    html = generate_html(metrics, generated_at)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard.html')
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Dashboard saved to: {output_path}")
    print(f"\n=== SUMMARY ===")
    print(f"Balance: ${metrics['balance_usd']:.2f}")
    print(f"Portfolio Value: ${metrics['portfolio_value_usd']:.2f}")
    print(f"Total Account: ${metrics['total_value_usd']:.2f}")
    print(f"Realized P&L: ${metrics['total_realized_pnl']:+.2f}")
    print(f"Open Positions: {len(metrics['open_market_positions'])}")
    print(f"Total Fills: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.0f}%")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
