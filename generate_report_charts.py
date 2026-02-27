"""Generate diagnostic charts for the trading bot analysis report."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

plt.style.use('seaborn-v0_8-whitegrid')

with open('/tmp/full_analysis_data.json', 'r') as f:
    data = json.load(f)

fills = data['fills']
settlements = data['settlements']
open_positions = data['open_positions']
balance = data['balance']

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
    elif 'KXTRUMP' in t or 'KXSENATE' in t or 'KXHOUSE' in t:
        return 'Politics'
    elif 'KXPGA' in t or 'KXGOLF' in t:
        return 'Golf'
    elif 'KXMV' in t or 'ESPORT' in t:
        return 'Esports'
    else:
        return 'Other'

# ============================================================
# Chart 1: P&L by Strategy (Settlements)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Settlement P&L by strategy
strat_pnl = defaultdict(lambda: {'won': 0, 'lost': 0, 'revenue': 0})
for s in settlements:
    ticker = s.get('market_ticker', s.get('ticker', ''))
    revenue = s.get('revenue', 0) / 100
    mt = classify_market(ticker)
    strat_pnl[mt]['revenue'] += revenue
    if revenue > 0:
        strat_pnl[mt]['won'] += 1
    else:
        strat_pnl[mt]['lost'] += 1

# Also compute cost basis for settled positions from fills
strat_cost = defaultdict(float)
settled_tickers = set()
for s in settlements:
    ticker = s.get('market_ticker', s.get('ticker', ''))
    settled_tickers.add(ticker)

for f_item in fills:
    ticker = f_item.get('ticker', '')
    if ticker in settled_tickers and f_item.get('action', '') == 'buy':
        price = f_item.get('yes_price', 0) if f_item.get('side') == 'yes' else f_item.get('no_price', 0)
        count = f_item.get('count', 0)
        cost = price * count / 100
        mt = classify_market(ticker)
        strat_cost[mt] += cost

strategies = sorted(strat_pnl.keys(), key=lambda x: strat_pnl[x]['revenue'], reverse=True)
revenues = [strat_pnl[s]['revenue'] for s in strategies]
costs = [strat_cost.get(s, 0) for s in strategies]
net_pnl = [strat_pnl[s]['revenue'] - strat_cost.get(s, 0) for s in strategies]

ax = axes[0, 0]
x = np.arange(len(strategies))
width = 0.35
bars1 = ax.bar(x - width/2, revenues, width, label='Settlement Revenue', color='#2ecc71', alpha=0.8)
bars2 = ax.bar(x + width/2, costs, width, label='Cost Basis', color='#e74c3c', alpha=0.8)
ax.set_xlabel('Strategy')
ax.set_ylabel('Dollars ($)')
ax.set_title('Settlement Revenue vs Cost Basis by Strategy')
ax.set_xticks(x)
ax.set_xticklabels(strategies, rotation=45, ha='right')
ax.legend()
ax.axhline(y=0, color='black', linewidth=0.5)

# ============================================================
# Chart 2: Win Rate by Strategy
# ============================================================
ax = axes[0, 1]
win_rates = []
labels = []
colors = []
for s in strategies:
    total = strat_pnl[s]['won'] + strat_pnl[s]['lost']
    if total > 0:
        wr = strat_pnl[s]['won'] / total * 100
        win_rates.append(wr)
        labels.append(f"{s}\n({strat_pnl[s]['won']}W/{strat_pnl[s]['lost']}L)")
        colors.append('#2ecc71' if wr >= 50 else '#e74c3c')

bars = ax.bar(range(len(win_rates)), win_rates, color=colors, alpha=0.8)
ax.set_ylabel('Win Rate (%)')
ax.set_title('Win Rate by Strategy (Settled Positions)')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='50% breakeven')
ax.legend()
for bar, wr in zip(bars, win_rates):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{wr:.0f}%', ha='center', va='bottom', fontweight='bold')

# ============================================================
# Chart 3: Portfolio Value Over Time (reconstructed from fills)
# ============================================================
ax = axes[1, 0]

# Reconstruct portfolio timeline
fill_timeline = []
for f_item in fills:
    ts = f_item.get('created_time', f_item.get('created_at', ''))
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            action = f_item.get('action', '')
            side = f_item.get('side', '')
            yes_price = f_item.get('yes_price', 0)
            no_price = f_item.get('no_price', 0)
            count = f_item.get('count', 0)
            price = yes_price if side == 'yes' else no_price
            cost = price * count / 100
            fill_timeline.append((dt, action, cost))
        except:
            pass

fill_timeline.sort(key=lambda x: x[0])

# Walk forward: start at $400, subtract buys, add sells
deposit = 400.0
cash = deposit
cash_history = [(fill_timeline[0][0] if fill_timeline else datetime.now(timezone.utc), deposit)]
for dt, action, cost in fill_timeline:
    if action == 'buy':
        cash -= cost
    else:
        cash += cost
    cash_history.append((dt, cash))

# Current state
cash_now = balance.get('balance', 0) / 100
pv_now = balance.get('portfolio_value', 0) / 100
total_now = cash_now + pv_now

times = [ch[0] for ch in cash_history]
cash_vals = [ch[1] for ch in cash_history]

# Estimate total portfolio (cash + unrealized) — scale from cash
# Since we know current cash and portfolio value
if cash_vals:
    total_estimated = []
    for cv in cash_vals:
        # Rough estimate: total = cash + (deposit - cash) as position value
        # Better: use the ratio of current portfolio/cash to scale
        invested = deposit - cv
        if invested > 0:
            # Scale position value proportionally
            total_est = cv + invested * (pv_now / max(1, deposit - cash_now))
        else:
            total_est = cv
        total_estimated.append(total_est)

ax.plot(times, total_estimated, 'r-', linewidth=2, label=f'Total Portfolio (est.)', alpha=0.9)
ax.fill_between(times, cash_vals, alpha=0.2, color='green', label='Available Cash')
ax.plot(times, cash_vals, 'g-', linewidth=1, alpha=0.5)
ax.axhline(y=deposit, color='gray', linestyle='--', linewidth=1, label=f'Deposited (${deposit:.0f})')
ax.set_xlabel('Time')
ax.set_ylabel('Value ($)')
ax.set_title('Portfolio Value Over Time')
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

# ============================================================
# Chart 4: Fill Distribution by Hour and Buy/Sell Ratio
# ============================================================
ax = axes[1, 1]

hourly_buys = defaultdict(int)
hourly_sells = defaultdict(int)
for f_item in fills:
    ts = f_item.get('created_time', f_item.get('created_at', ''))
    action = f_item.get('action', '')
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            if action == 'buy':
                hourly_buys[dt.hour] += 1
            else:
                hourly_sells[dt.hour] += 1
        except:
            pass

hours = range(24)
buys = [hourly_buys.get(h, 0) for h in hours]
sells = [hourly_sells.get(h, 0) for h in hours]

ax.bar(hours, buys, color='#e74c3c', alpha=0.7, label='Buys')
ax.bar(hours, [-s for s in sells], color='#2ecc71', alpha=0.7, label='Sells')
ax.set_xlabel('Hour (UTC)')
ax.set_ylabel('Fill Count')
ax.set_title('Buy vs Sell Activity by Hour')
ax.legend()
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xticks(range(0, 24, 2))

plt.tight_layout()
plt.savefig('/home/ubuntu/kalshi-ai-trading-bot/report_charts.png', dpi=150, bbox_inches='tight')
print("Charts saved to report_charts.png")

# ============================================================
# Chart 5: Detailed Weather Analysis (separate figure)
# ============================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

# Weather: city breakdown
city_stats = defaultdict(lambda: {'buys': 0, 'cost': 0, 'revenue': 0, 'won': 0, 'lost': 0})
for f_item in fills:
    ticker = f_item.get('ticker', '').upper()
    if 'KXHIGH' not in ticker:
        continue
    city = ticker.split('-')[0].replace('KXHIGH', '')
    action = f_item.get('action', '')
    price = f_item.get('yes_price', 0) if f_item.get('side') == 'yes' else f_item.get('no_price', 0)
    count = f_item.get('count', 0)
    cost = price * count / 100
    if action == 'buy':
        city_stats[city]['buys'] += 1
        city_stats[city]['cost'] += cost

for s in settlements:
    ticker = s.get('market_ticker', s.get('ticker', '')).upper()
    if 'KXHIGH' not in ticker:
        continue
    city = ticker.split('-')[0].replace('KXHIGH', '')
    revenue = s.get('revenue', 0) / 100
    city_stats[city]['revenue'] += revenue
    if revenue > 0:
        city_stats[city]['won'] += 1
    else:
        city_stats[city]['lost'] += 1

cities = sorted(city_stats.keys(), key=lambda x: city_stats[x]['cost'], reverse=True)

# City cost vs revenue
ax = axes2[0]
city_costs = [city_stats[c]['cost'] for c in cities]
city_revs = [city_stats[c]['revenue'] for c in cities]
x = np.arange(len(cities))
width = 0.35
ax.bar(x - width/2, city_costs, width, label='Cost Basis', color='#e74c3c', alpha=0.8)
ax.bar(x + width/2, city_revs, width, label='Revenue', color='#2ecc71', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(cities, rotation=45, ha='right')
ax.set_ylabel('Dollars ($)')
ax.set_title('Weather: Cost vs Revenue by City')
ax.legend()

# City win rates
ax = axes2[1]
city_wr = []
city_labels = []
for c in cities:
    total = city_stats[c]['won'] + city_stats[c]['lost']
    if total > 0:
        wr = city_stats[c]['won'] / total * 100
        city_wr.append(wr)
        city_labels.append(f"{c}\n({city_stats[c]['won']}W/{city_stats[c]['lost']}L)")
    else:
        city_wr.append(0)
        city_labels.append(f"{c}\n(open)")

colors = ['#2ecc71' if wr >= 50 else '#e74c3c' for wr in city_wr]
ax.bar(range(len(city_wr)), city_wr, color=colors, alpha=0.8)
ax.set_xticks(range(len(city_labels)))
ax.set_xticklabels(city_labels, rotation=45, ha='right')
ax.set_ylabel('Win Rate (%)')
ax.set_title('Weather: Win Rate by City')
ax.axhline(y=50, color='gray', linestyle='--', linewidth=1)

# Position size distribution
ax = axes2[2]
buy_costs = []
for f_item in fills:
    if f_item.get('action') == 'buy':
        price = f_item.get('yes_price', 0) if f_item.get('side') == 'yes' else f_item.get('no_price', 0)
        count = f_item.get('count', 0)
        cost = price * count / 100
        if cost > 0:
            buy_costs.append(cost)

ax.hist(buy_costs, bins=30, color='#3498db', alpha=0.8, edgecolor='white')
ax.set_xlabel('Position Size ($)')
ax.set_ylabel('Count')
ax.set_title('Position Size Distribution (All Buys)')
ax.axvline(x=np.median(buy_costs), color='red', linestyle='--', label=f'Median: ${np.median(buy_costs):.2f}')
ax.axvline(x=np.mean(buy_costs), color='orange', linestyle='--', label=f'Mean: ${np.mean(buy_costs):.2f}')
ax.legend()

plt.tight_layout()
plt.savefig('/home/ubuntu/kalshi-ai-trading-bot/report_weather_charts.png', dpi=150, bbox_inches='tight')
print("Weather charts saved to report_weather_charts.png")
