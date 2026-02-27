"""
Deep analysis of trading bot performance.
Analyzes fills, orders, positions, settlements by strategy, market type, time, and identifies patterns.
"""
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import re

# Load data
with open('/tmp/full_analysis_data.json', 'r') as f:
    data = json.load(f)

fills = data['fills']
orders = data['orders']
open_positions = data['open_positions']
settled_positions = data['settled_positions']
settlements = data['settlements']
balance = data['balance']

cash = balance.get('balance', 0) / 100
pv = balance.get('portfolio_value', 0) / 100
total = cash + pv
deposit = 400.0

print("=" * 80)
print("DEEP PERFORMANCE ANALYSIS — Kalshi AI Trading Bot")
print("=" * 80)
print(f"Analysis Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
print(f"Account: Cash ${cash:.2f} + Portfolio ${pv:.2f} = Total ${total:.2f}")
print(f"Deposit: ${deposit:.2f}")
print(f"Total P&L: ${total - deposit:.2f} ({(total/deposit - 1)*100:.1f}%)")
print()

# ============================================================
# 1. FILL ANALYSIS
# ============================================================
print("=" * 80)
print("1. FILL ANALYSIS")
print("=" * 80)

# Classify fills by market type (ticker prefix)
def classify_market(ticker):
    t = ticker.upper()
    if 'KXHIGH' in t or 'KXLOW' in t or 'KXRAIN' in t or 'KXSNOW' in t or 'KXWIND' in t:
        return 'Weather'
    elif 'KXNBA' in t:
        return 'NBA'
    elif 'KXNFL' in t:
        return 'NFL'
    elif 'KXNHL' in t:
        return 'NHL'
    elif 'KXSOCCER' in t or 'KXUCL' in t or 'KXEPL' in t:
        return 'Soccer'
    elif 'KXCPI' in t or 'KXGDP' in t or 'KXJOBLESS' in t or 'KXUNEMPLOY' in t or 'KXFED' in t:
        return 'Economics'
    elif 'KXGAS' in t or 'KXOIL' in t:
        return 'Gas/Oil'
    elif 'KXTRUMP' in t or 'KXSENATE' in t or 'KXHOUSE' in t or 'KXBIDEN' in t:
        return 'Politics'
    elif 'KXPGA' in t or 'KXGOLF' in t:
        return 'Golf'
    elif 'KXMV' in t or 'ESPORT' in t:
        return 'Esports'
    elif 'KXBTC' in t or 'KXETH' in t or 'KXCRYPTO' in t:
        return 'Crypto'
    elif 'KXSP500' in t or 'KXNASDAQ' in t or 'KXDOW' in t or 'KXSTOCK' in t:
        return 'Stocks'
    else:
        return 'Other'

# Parse fills
fill_data = []
for f_item in fills:
    ticker = f_item.get('ticker', '')
    side = f_item.get('side', '')
    action = f_item.get('action', f_item.get('type', ''))
    yes_price = f_item.get('yes_price', 0)
    no_price = f_item.get('no_price', 0)
    count = f_item.get('count', 0)
    created = f_item.get('created_time', f_item.get('created_at', ''))
    is_taker = f_item.get('is_taker', None)
    order_id = f_item.get('order_id', '')
    
    # Price in cents
    price = yes_price if side == 'yes' else no_price
    cost = price * count / 100  # in dollars
    
    market_type = classify_market(ticker)
    
    fill_data.append({
        'ticker': ticker,
        'side': side,
        'action': action,
        'price': price,
        'count': count,
        'cost': cost,
        'market_type': market_type,
        'created': created,
        'is_taker': is_taker,
        'order_id': order_id
    })

# Summary by market type
print("\n--- Fills by Market Type ---")
type_stats = defaultdict(lambda: {'count': 0, 'buy_count': 0, 'sell_count': 0, 'buy_cost': 0, 'sell_cost': 0, 'tickers': set()})
for fd in fill_data:
    mt = fd['market_type']
    type_stats[mt]['count'] += 1
    type_stats[mt]['tickers'].add(fd['ticker'])
    if fd['action'] == 'buy':
        type_stats[mt]['buy_count'] += 1
        type_stats[mt]['buy_cost'] += fd['cost']
    else:
        type_stats[mt]['sell_count'] += 1
        type_stats[mt]['sell_cost'] += fd['cost']

print(f"{'Market Type':<15} {'Fills':>6} {'Buys':>6} {'Sells':>6} {'Buy $':>10} {'Sell $':>10} {'Net $':>10} {'Markets':>8}")
print("-" * 80)
for mt in sorted(type_stats.keys(), key=lambda x: type_stats[x]['buy_cost'], reverse=True):
    s = type_stats[mt]
    net = s['sell_cost'] - s['buy_cost']
    print(f"{mt:<15} {s['count']:>6} {s['buy_count']:>6} {s['sell_count']:>6} ${s['buy_cost']:>9.2f} ${s['sell_cost']:>9.2f} ${net:>9.2f} {len(s['tickers']):>8}")

# ============================================================
# 2. SETTLEMENT ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("2. SETTLEMENT ANALYSIS (Realized P&L)")
print("=" * 80)

settled_pnl = defaultdict(lambda: {'won': 0, 'lost': 0, 'pnl': 0, 'revenue': 0, 'cost': 0, 'tickers': set()})
for s in settlements:
    ticker = s.get('market_ticker', s.get('ticker', ''))
    revenue = s.get('revenue', 0) / 100  # cents to dollars
    settled_at = s.get('settled_time', s.get('settled_at', ''))
    yes_price = s.get('yes_price', 0)
    no_price = s.get('no_price', 0)
    market_result = s.get('market_result', '')
    
    mt = classify_market(ticker)
    settled_pnl[mt]['tickers'].add(ticker)
    settled_pnl[mt]['revenue'] += revenue
    
    if revenue > 0:
        settled_pnl[mt]['won'] += 1
    else:
        settled_pnl[mt]['lost'] += 1

print(f"\n{'Market Type':<15} {'Won':>5} {'Lost':>5} {'Win%':>6} {'Revenue':>10} {'Markets':>8}")
print("-" * 60)
total_won = 0
total_lost = 0
total_revenue = 0
for mt in sorted(settled_pnl.keys(), key=lambda x: settled_pnl[x]['revenue'], reverse=True):
    s = settled_pnl[mt]
    total_trades = s['won'] + s['lost']
    win_pct = (s['won'] / total_trades * 100) if total_trades > 0 else 0
    print(f"{mt:<15} {s['won']:>5} {s['lost']:>5} {win_pct:>5.1f}% ${s['revenue']:>9.2f} {len(s['tickers']):>8}")
    total_won += s['won']
    total_lost += s['lost']
    total_revenue += s['revenue']

total_trades_all = total_won + total_lost
win_pct_all = (total_won / total_trades_all * 100) if total_trades_all > 0 else 0
print("-" * 60)
print(f"{'TOTAL':<15} {total_won:>5} {total_lost:>5} {win_pct_all:>5.1f}% ${total_revenue:>9.2f}")

# ============================================================
# 3. OPEN POSITIONS ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("3. OPEN POSITIONS ANALYSIS")
print("=" * 80)

pos_by_type = defaultdict(lambda: {'count': 0, 'exposure': 0, 'market_value': 0, 'tickers': []})
for p in open_positions:
    ticker = p.get('ticker', p.get('market_ticker', ''))
    position = p.get('position', 0)
    market_exposure = p.get('market_exposure', 0) / 100
    total_traded = p.get('total_traded', 0) / 100
    resting_orders_count = p.get('resting_orders_count', 0)
    
    mt = classify_market(ticker)
    pos_by_type[mt]['count'] += 1
    pos_by_type[mt]['exposure'] += market_exposure
    pos_by_type[mt]['tickers'].append(ticker)

print(f"\n{'Market Type':<15} {'Positions':>10} {'Exposure':>12} {'Tickers'}")
print("-" * 80)
for mt in sorted(pos_by_type.keys(), key=lambda x: pos_by_type[x]['exposure'], reverse=True):
    p = pos_by_type[mt]
    tickers_str = ', '.join(p['tickers'][:3])
    if len(p['tickers']) > 3:
        tickers_str += f" +{len(p['tickers'])-3} more"
    print(f"{mt:<15} {p['count']:>10} ${p['exposure']:>10.2f}   {tickers_str}")

# ============================================================
# 4. ORDER ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("4. ORDER ANALYSIS")
print("=" * 80)

order_stats = defaultdict(lambda: {'total': 0, 'filled': 0, 'cancelled': 0, 'pending': 0, 'expired': 0})
for o in orders:
    ticker = o.get('ticker', '')
    status = o.get('status', '')
    mt = classify_market(ticker)
    order_stats[mt]['total'] += 1
    if status in ('executed', 'filled'):
        order_stats[mt]['filled'] += 1
    elif status == 'canceled':
        order_stats[mt]['cancelled'] += 1
    elif status in ('pending', 'resting'):
        order_stats[mt]['pending'] += 1
    else:
        order_stats[mt]['expired'] += 1

print(f"\n{'Market Type':<15} {'Total':>7} {'Filled':>7} {'Cancel':>7} {'Pending':>8} {'Other':>7} {'Fill%':>7}")
print("-" * 65)
for mt in sorted(order_stats.keys(), key=lambda x: order_stats[x]['total'], reverse=True):
    o = order_stats[mt]
    fill_pct = (o['filled'] / o['total'] * 100) if o['total'] > 0 else 0
    print(f"{mt:<15} {o['total']:>7} {o['filled']:>7} {o['cancelled']:>7} {o['pending']:>8} {o['expired']:>7} {fill_pct:>6.1f}%")

# ============================================================
# 5. TEMPORAL ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("5. TEMPORAL ANALYSIS")
print("=" * 80)

hourly_fills = defaultdict(int)
daily_fills = defaultdict(lambda: {'count': 0, 'buy_cost': 0, 'sell_cost': 0})
for fd in fill_data:
    ts = fd['created']
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            hourly_fills[dt.hour] += 1
            day = dt.strftime('%Y-%m-%d')
            daily_fills[day]['count'] += 1
            if fd['action'] == 'buy':
                daily_fills[day]['buy_cost'] += fd['cost']
            else:
                daily_fills[day]['sell_cost'] += fd['cost']
        except:
            pass

print("\n--- Hourly Distribution (UTC) ---")
for h in sorted(hourly_fills.keys()):
    bar = '#' * (hourly_fills[h] // 2)
    print(f"  {h:02d}:00  {hourly_fills[h]:>4} fills  {bar}")

print("\n--- Daily Summary ---")
print(f"{'Date':<12} {'Fills':>6} {'Buy $':>10} {'Sell $':>10} {'Net $':>10}")
print("-" * 55)
for day in sorted(daily_fills.keys()):
    d = daily_fills[day]
    net = d['sell_cost'] - d['buy_cost']
    print(f"{day:<12} {d['count']:>6} ${d['buy_cost']:>9.2f} ${d['sell_cost']:>9.2f} ${net:>9.2f}")

# ============================================================
# 6. FEES ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("6. FEES ANALYSIS")
print("=" * 80)

total_fees = 0
for f_item in fills:
    fee = f_item.get('fee', 0)
    if isinstance(fee, (int, float)):
        total_fees += fee

total_fees_dollars = total_fees / 100
print(f"Total Fees Paid: ${total_fees_dollars:.2f}")
print(f"Fees as % of Deposit: {total_fees_dollars/deposit*100:.1f}%")
print(f"Avg Fee per Fill: ${total_fees_dollars/len(fills):.4f}" if fills else "N/A")

# ============================================================
# 7. WORST POSITIONS
# ============================================================
print("\n" + "=" * 80)
print("7. WORST SETTLED POSITIONS (Biggest Losses)")
print("=" * 80)

# Sort settlements by revenue (ascending = worst first)
sorted_settlements = sorted(settlements, key=lambda x: x.get('revenue', 0))
print(f"\n{'Ticker':<55} {'Revenue':>10} {'Type':<12}")
print("-" * 80)
for s in sorted_settlements[:15]:
    ticker = s.get('market_ticker', s.get('ticker', ''))
    revenue = s.get('revenue', 0) / 100
    mt = classify_market(ticker)
    print(f"{ticker:<55} ${revenue:>9.2f} {mt:<12}")

print("\n--- Best Settled Positions ---")
for s in sorted_settlements[-10:]:
    ticker = s.get('market_ticker', s.get('ticker', ''))
    revenue = s.get('revenue', 0) / 100
    mt = classify_market(ticker)
    print(f"{ticker:<55} ${revenue:>9.2f} {mt:<12}")

# ============================================================
# 8. TAKER vs MAKER ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("8. TAKER vs MAKER ANALYSIS")
print("=" * 80)

taker_count = sum(1 for fd in fill_data if fd['is_taker'] == True)
maker_count = sum(1 for fd in fill_data if fd['is_taker'] == False)
unknown_count = len(fill_data) - taker_count - maker_count
print(f"Taker fills: {taker_count} ({taker_count/len(fill_data)*100:.1f}%)")
print(f"Maker fills: {maker_count} ({maker_count/len(fill_data)*100:.1f}%)")
print(f"Unknown: {unknown_count}")

# ============================================================
# 9. POSITION SIZE ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("9. POSITION SIZE ANALYSIS")
print("=" * 80)

costs = [fd['cost'] for fd in fill_data if fd['action'] == 'buy' and fd['cost'] > 0]
if costs:
    costs.sort()
    print(f"Total buy fills: {len(costs)}")
    print(f"Min position: ${min(costs):.2f}")
    print(f"Max position: ${max(costs):.2f}")
    print(f"Avg position: ${sum(costs)/len(costs):.2f}")
    print(f"Median position: ${costs[len(costs)//2]:.2f}")
    print(f"Total deployed: ${sum(costs):.2f}")
    
    # Distribution
    buckets = {'$0-1': 0, '$1-5': 0, '$5-10': 0, '$10-20': 0, '$20-50': 0, '$50+': 0}
    for c in costs:
        if c < 1: buckets['$0-1'] += 1
        elif c < 5: buckets['$1-5'] += 1
        elif c < 10: buckets['$5-10'] += 1
        elif c < 20: buckets['$10-20'] += 1
        elif c < 50: buckets['$20-50'] += 1
        else: buckets['$50+'] += 1
    print("\nPosition Size Distribution:")
    for bucket, count in buckets.items():
        bar = '#' * count
        print(f"  {bucket:<8} {count:>4}  {bar}")

# ============================================================
# 10. RAW DATA DUMP FOR FURTHER ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("10. KEY FINDINGS SUMMARY")
print("=" * 80)

print(f"""
ACCOUNT STATUS:
  Starting Deposit:  $400.00
  Current Total:     ${total:.2f}
  Total P&L:         ${total - deposit:.2f} ({(total/deposit-1)*100:.1f}%)
  Cash Available:    ${cash:.2f}
  Portfolio Value:   ${pv:.2f}
  
ACTIVITY:
  Total Fills:       {len(fills)}
  Total Orders:      {len(orders)}
  Open Positions:    {len(open_positions)}
  Settled Positions: {len(settled_positions)}
  Settlements:       {len(settlements)}
  Total Fees:        ${total_fees_dollars:.2f}
  
STRATEGY BREAKDOWN:
  Dominant Strategy: Weather ({type_stats.get('Weather', {}).get('count', 0)} fills)
  Most Positions:    Weather ({pos_by_type.get('Weather', {}).get('count', 0)} open)
  
RISK METRICS:
  Max Drawdown:      ${deposit - total:.2f} ({(1 - total/deposit)*100:.1f}% from deposit)
  Taker Rate:        {taker_count/len(fill_data)*100:.1f}%
""")
