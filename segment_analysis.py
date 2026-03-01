#!/usr/bin/env python3
"""
Deep analysis of portfolio curve segments:
- What causes each gradual decline?
- What causes each sharp jump?
- Fee drag vs losing positions vs capital deployment
"""
import os, sys, json, asyncio
from datetime import datetime, timezone
from dateutil.parser import isoparse
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, '.')
from src.clients.kalshi_client import KalshiClient

async def main():
    client = KalshiClient()

    # Pull all fills
    print("=== PULLING ALL FILLS ===")
    all_fills = []
    cursor = None
    while True:
        params = {'limit': 1000}
        if cursor:
            params['cursor'] = cursor
        resp = await client._make_authenticated_request('GET', '/trade-api/v2/portfolio/fills', params=params)
        fills = resp.get('fills', [])
        if not fills:
            break
        all_fills.extend(fills)
        cursor = resp.get('cursor')
        if not cursor:
            break
    print(f"Total fills: {len(all_fills)}")

    # Pull all settlements
    print("\n=== PULLING ALL SETTLEMENTS ===")
    all_settlements = []
    cursor = None
    while True:
        params = {'limit': 1000}
        if cursor:
            params['cursor'] = cursor
        resp = await client._make_authenticated_request('GET', '/trade-api/v2/portfolio/settlements', params=params)
        settlements = resp.get('settlements', [])
        if not settlements:
            break
        all_settlements.extend(settlements)
        cursor = resp.get('cursor')
        if not cursor:
            break
    print(f"Total settlements: {len(all_settlements)}")

    # Get current balance
    bal = await client.get_balance()
    cash = bal.get('balance', 0) / 100.0
    portfolio_value = bal.get('portfolio_value', 0) / 100.0
    print(f"\nCurrent cash: ${cash:.2f}, Portfolio value: ${portfolio_value:.2f}, Total: ${cash + portfolio_value:.2f}")

    # Parse fills with timestamps
    parsed_fills = []
    for f in all_fills:
        ts_str = f.get('created_time', '')
        try:
            dt = isoparse(ts_str)
            ts = int(dt.timestamp())
        except:
            continue
        count = f.get('count', 0)
        price = f.get('yes_price', 0) or f.get('no_price', 0)
        cost = (count * price) / 100.0
        fee_raw = float(f.get('fee', 0))
        fee = fee_raw / 100.0 if fee_raw > 1.0 else fee_raw
        action = f.get('action', 'buy')
        side = f.get('side', 'yes')
        ticker = f.get('ticker', '')
        is_taker = f.get('is_taker', True)
        parsed_fills.append({
            'ts': ts, 'dt': dt, 'action': action, 'side': side,
            'count': count, 'price': price, 'cost': cost, 'fee': fee,
            'ticker': ticker, 'is_taker': is_taker
        })
    parsed_fills.sort(key=lambda x: x['ts'])

    # Parse settlements
    parsed_settlements = []
    for s in all_settlements:
        ts_str = s.get('settled_time', '')
        try:
            dt = isoparse(ts_str)
            ts = int(dt.timestamp())
        except:
            continue
        revenue = s.get('revenue', 0) / 100.0
        yes_cost = s.get('yes_total_cost', 0) / 100.0
        no_cost = s.get('no_total_cost', 0) / 100.0
        total_cost = yes_cost + no_cost
        net_pnl = revenue - total_cost
        ticker = s.get('ticker', '')
        market_ticker = s.get('market_ticker', ticker)
        result = s.get('settlement_result', '')
        parsed_settlements.append({
            'ts': ts, 'dt': dt, 'net_pnl': net_pnl, 'revenue': revenue,
            'total_cost': total_cost, 'ticker': ticker, 'market': market_ticker,
            'result': result
        })
    parsed_settlements.sort(key=lambda x: x['ts'])

    # === DAILY BREAKDOWN ===
    print("\n" + "="*80)
    print("DAILY P&L BREAKDOWN")
    print("="*80)

    daily = defaultdict(lambda: {'fees': 0, 'buy_cost': 0, 'buy_count': 0, 'sell_cost': 0,
                                  'settlement_pnl': 0, 'sett_wins': 0, 'sett_losses': 0,
                                  'taker_fills': 0, 'maker_fills': 0, 'taker_fees': 0, 'maker_fees': 0})

    for f in parsed_fills:
        day = f['dt'].strftime('%Y-%m-%d')
        daily[day]['fees'] += f['fee']
        if f['action'] == 'buy':
            daily[day]['buy_cost'] += f['cost']
            daily[day]['buy_count'] += f['count']
        else:
            daily[day]['sell_cost'] += f['cost']
        if f.get('is_taker'):
            daily[day]['taker_fills'] += 1
            daily[day]['taker_fees'] += f['fee']
        else:
            daily[day]['maker_fills'] += 1
            daily[day]['maker_fees'] += f['fee']

    for s in parsed_settlements:
        day = s['dt'].strftime('%Y-%m-%d')
        daily[day]['settlement_pnl'] += s['net_pnl']
        if s['net_pnl'] > 0:
            daily[day]['sett_wins'] += 1
        elif s['net_pnl'] < 0:
            daily[day]['sett_losses'] += 1

    print(f"\n{'Date':<12} {'Fees':>8} {'Buys':>8} {'#Contracts':>11} {'Sett P&L':>10} {'W/L':>6} {'Taker%':>8} {'Net Day':>10}")
    print("-" * 85)
    for day in sorted(daily.keys()):
        d = daily[day]
        total_fills = d['taker_fills'] + d['maker_fills']
        taker_pct = d['taker_fills'] / total_fills * 100 if total_fills > 0 else 0
        net_day = d['settlement_pnl'] - d['fees']
        wl = f"{d['sett_wins']}/{d['sett_losses']}"
        print(f"{day:<12} ${d['fees']:>7.2f} ${d['buy_cost']:>7.0f} {d['buy_count']:>11} ${d['settlement_pnl']:>+9.2f} {wl:>6} {taker_pct:>7.0f}% ${net_day:>+9.2f}")

    # === SETTLEMENT BATCH ANALYSIS ===
    print("\n" + "="*80)
    print("SETTLEMENT BATCHES (the jumps in the curve)")
    print("="*80)

    sett_times = sorted(set(s['ts'] for s in parsed_settlements))
    batches = []
    if sett_times:
        batch = [sett_times[0]]
        for i in range(1, len(sett_times)):
            if sett_times[i] - sett_times[i-1] > 7200:
                batches.append(batch)
                batch = [sett_times[i]]
            else:
                batch.append(sett_times[i])
        batches.append(batch)

    for i, batch in enumerate(batches):
        batch_dt = datetime.fromtimestamp(batch[0], tz=timezone.utc)
        batch_setts = [s for s in parsed_settlements if s['ts'] >= batch[0] and s['ts'] <= batch[-1]]
        batch_pnl = sum(s['net_pnl'] for s in batch_setts)
        batch_wins = sum(1 for s in batch_setts if s['net_pnl'] > 0)
        batch_losses = sum(1 for s in batch_setts if s['net_pnl'] <= 0)
        batch_cost = sum(s['total_cost'] for s in batch_setts)
        roi = batch_pnl / batch_cost * 100 if batch_cost > 0 else 0
        marker = "+++" if batch_pnl > 50 else "---" if batch_pnl < -50 else "   "
        print(f"  {marker} Batch {i+1}: {batch_dt.strftime('%b %d %H:%M')} | {len(batch_setts)} sett | P&L: ${batch_pnl:+.2f} | ROI: {roi:+.0f}% | W:{batch_wins} L:{batch_losses} | Cost: ${batch_cost:.0f}")

    # === BIG JUMPS DETAIL ===
    print("\n" + "="*80)
    print("BIG POSITIVE JUMPS (what worked)")
    print("="*80)

    for i, batch in enumerate(batches):
        batch_setts = [s for s in parsed_settlements if s['ts'] >= batch[0] and s['ts'] <= batch[-1]]
        batch_pnl = sum(s['net_pnl'] for s in batch_setts)
        if batch_pnl > 20:
            batch_dt = datetime.fromtimestamp(batch[0], tz=timezone.utc)
            print(f"\n*** JUMP: {batch_dt.strftime('%b %d %H:%M')} | P&L: ${batch_pnl:+.2f} ***")
            for s in sorted(batch_setts, key=lambda x: x['net_pnl'], reverse=True):
                if s['net_pnl'] != 0:
                    print(f"  {s['ticker']}: ${s['net_pnl']:+.2f} (cost: ${s['total_cost']:.2f}, rev: ${s['revenue']:.2f})")

    # === BIG DECLINES DETAIL ===
    print("\n" + "="*80)
    print("BIG NEGATIVE DROPS (what failed)")
    print("="*80)

    for i, batch in enumerate(batches):
        batch_setts = [s for s in parsed_settlements if s['ts'] >= batch[0] and s['ts'] <= batch[-1]]
        batch_pnl = sum(s['net_pnl'] for s in batch_setts)
        if batch_pnl < -20:
            batch_dt = datetime.fromtimestamp(batch[0], tz=timezone.utc)
            print(f"\n*** DROP: {batch_dt.strftime('%b %d %H:%M')} | P&L: ${batch_pnl:+.2f} ***")
            for s in sorted(batch_setts, key=lambda x: x['net_pnl']):
                if s['net_pnl'] != 0:
                    print(f"  {s['ticker']}: ${s['net_pnl']:+.2f} (cost: ${s['total_cost']:.2f}, rev: ${s['revenue']:.2f})")

    # === FILL PRICE ANALYSIS ===
    print("\n" + "="*80)
    print("FILL PRICE DISTRIBUTION (what prices are we buying at?)")
    print("="*80)

    buy_fills = [f for f in parsed_fills if f['action'] == 'buy']
    price_buckets = {'0-5c': [], '5-15c': [], '15-30c': [], '30-50c': [], '50-75c': [], '75-100c': []}
    for f in buy_fills:
        p = f['price']
        if p <= 5: price_buckets['0-5c'].append(f)
        elif p <= 15: price_buckets['5-15c'].append(f)
        elif p <= 30: price_buckets['15-30c'].append(f)
        elif p <= 50: price_buckets['30-50c'].append(f)
        elif p <= 75: price_buckets['50-75c'].append(f)
        else: price_buckets['75-100c'].append(f)

    print(f"\n{'Price Range':<12} {'#Fills':>8} {'#Contracts':>12} {'Total Cost':>12} {'Avg Price':>10} {'Taker%':>8}")
    print("-" * 70)
    for bucket, fills in price_buckets.items():
        if fills:
            total_contracts = sum(f['count'] for f in fills)
            total_cost = sum(f['cost'] for f in fills)
            avg_price = sum(f['price'] * f['count'] for f in fills) / total_contracts if total_contracts > 0 else 0
            taker_pct = sum(1 for f in fills if f.get('is_taker')) / len(fills) * 100
            print(f"{bucket:<12} {len(fills):>8} {total_contracts:>12} ${total_cost:>11.2f} {avg_price:>9.1f}c {taker_pct:>7.0f}%")

    # === SETTLEMENT WIN RATE BY ENTRY PRICE ===
    print("\n" + "="*80)
    print("SETTLEMENT WIN RATE BY ENTRY PRICE")
    print("="*80)

    for s in parsed_settlements:
        ticker_fills = [f for f in parsed_fills if f['ticker'] == s['ticker'] and f['action'] == 'buy']
        if ticker_fills:
            avg_entry = sum(f['price'] * f['count'] for f in ticker_fills) / sum(f['count'] for f in ticker_fills)
            s['avg_entry_price'] = avg_entry
        else:
            s['avg_entry_price'] = None

    price_win_rate = {'0-5c': {'wins': 0, 'losses': 0, 'pnl': 0, 'cost': 0},
                      '5-15c': {'wins': 0, 'losses': 0, 'pnl': 0, 'cost': 0},
                      '15-30c': {'wins': 0, 'losses': 0, 'pnl': 0, 'cost': 0},
                      '30-50c': {'wins': 0, 'losses': 0, 'pnl': 0, 'cost': 0},
                      '50-75c': {'wins': 0, 'losses': 0, 'pnl': 0, 'cost': 0},
                      '75-100c': {'wins': 0, 'losses': 0, 'pnl': 0, 'cost': 0}}

    for s in parsed_settlements:
        if s['avg_entry_price'] is None:
            continue
        p = s['avg_entry_price']
        if p <= 5: bucket = '0-5c'
        elif p <= 15: bucket = '5-15c'
        elif p <= 30: bucket = '15-30c'
        elif p <= 50: bucket = '30-50c'
        elif p <= 75: bucket = '50-75c'
        else: bucket = '75-100c'

        if s['net_pnl'] > 0:
            price_win_rate[bucket]['wins'] += 1
        else:
            price_win_rate[bucket]['losses'] += 1
        price_win_rate[bucket]['pnl'] += s['net_pnl']
        price_win_rate[bucket]['cost'] += s['total_cost']

    print(f"\n{'Price Range':<12} {'Wins':>6} {'Losses':>8} {'Win Rate':>10} {'Net P&L':>10} {'ROI':>8}")
    print("-" * 60)
    for bucket, data in price_win_rate.items():
        total = data['wins'] + data['losses']
        wr = data['wins'] / total * 100 if total > 0 else 0
        roi = data['pnl'] / data['cost'] * 100 if data['cost'] > 0 else 0
        print(f"{bucket:<12} {data['wins']:>6} {data['losses']:>8} {wr:>9.1f}% ${data['pnl']:>+9.2f} {roi:>+7.0f}%")

    # === BETWEEN-SETTLEMENT FEE DRAG ===
    print("\n" + "="*80)
    print("FEE DRAG BETWEEN SETTLEMENTS (the gradual decline)")
    print("="*80)

    total_fees = sum(f['fee'] for f in parsed_fills)
    total_taker_fees = sum(f['fee'] for f in parsed_fills if f.get('is_taker'))
    total_maker_fees = sum(f['fee'] for f in parsed_fills if not f.get('is_taker'))
    taker_count = sum(1 for f in parsed_fills if f.get('is_taker'))
    maker_count = sum(1 for f in parsed_fills if not f.get('is_taker'))

    print(f"\nTotal fees: ${total_fees:.2f} ({total_fees/400*100:.1f}% of deposit)")
    print(f"  Taker fees: ${total_taker_fees:.2f} from {taker_count} fills ({taker_count/(taker_count+maker_count)*100:.0f}% of fills)")
    print(f"  Maker fees: ${total_maker_fees:.2f} from {maker_count} fills ({maker_count/(taker_count+maker_count)*100:.0f}% of fills)")
    print(f"  Avg fee per fill: ${total_fees/len(parsed_fills):.4f}")
    print(f"  Avg taker fee: ${total_taker_fees/taker_count:.4f}" if taker_count > 0 else "")
    print(f"  Avg maker fee: ${total_maker_fees/maker_count:.4f}" if maker_count > 0 else "")

    # === TICKER-LEVEL P&L ===
    print("\n" + "="*80)
    print("TOP 10 BIGGEST LOSERS (by ticker)")
    print("="*80)

    ticker_pnl = defaultdict(lambda: {'pnl': 0, 'cost': 0, 'fees': 0, 'count': 0})
    for s in parsed_settlements:
        ticker_pnl[s['ticker']]['pnl'] += s['net_pnl']
        ticker_pnl[s['ticker']]['cost'] += s['total_cost']
    for f in parsed_fills:
        ticker_pnl[f['ticker']]['fees'] += f['fee']
        if f['action'] == 'buy':
            ticker_pnl[f['ticker']]['count'] += f['count']

    sorted_losers = sorted(ticker_pnl.items(), key=lambda x: x[1]['pnl'])
    print(f"\n{'Ticker':<40} {'P&L':>10} {'Cost':>10} {'Fees':>8} {'#Contracts':>12}")
    print("-" * 85)
    for ticker, data in sorted_losers[:15]:
        print(f"{ticker:<40} ${data['pnl']:>+9.2f} ${data['cost']:>9.2f} ${data['fees']:>7.2f} {data['count']:>12}")

    print("\n" + "="*80)
    print("TOP 10 BIGGEST WINNERS (by ticker)")
    print("="*80)
    sorted_winners = sorted(ticker_pnl.items(), key=lambda x: x[1]['pnl'], reverse=True)
    print(f"\n{'Ticker':<40} {'P&L':>10} {'Cost':>10} {'Fees':>8} {'#Contracts':>12}")
    print("-" * 85)
    for ticker, data in sorted_winners[:15]:
        print(f"{ticker:<40} ${data['pnl']:>+9.2f} ${data['cost']:>9.2f} ${data['fees']:>7.2f} {data['count']:>12}")

    # === CATEGORY ANALYSIS ===
    print("\n" + "="*80)
    print("P&L BY MARKET CATEGORY")
    print("="*80)

    cat_pnl = defaultdict(lambda: {'pnl': 0, 'cost': 0, 'fees': 0, 'trades': 0})
    for ticker, data in ticker_pnl.items():
        if 'KXHIGH' in ticker or 'KXLOW' in ticker:
            cat = 'Weather'
        elif 'KXCPI' in ticker or 'KXGDP' in ticker or 'KXJOB' in ticker or 'KXUNRATE' in ticker:
            cat = 'Economics'
        elif 'KXGAS' in ticker:
            cat = 'Gas'
        elif 'NBA' in ticker or 'NFL' in ticker:
            cat = 'Sports'
        elif 'FLU' in ticker or 'KXFLU' in ticker:
            cat = 'Flu'
        else:
            cat = 'Other'
        cat_pnl[cat]['pnl'] += data['pnl']
        cat_pnl[cat]['cost'] += data['cost']
        cat_pnl[cat]['fees'] += data['fees']
        cat_pnl[cat]['trades'] += data['count']

    print(f"\n{'Category':<15} {'P&L':>10} {'Cost':>10} {'Fees':>8} {'ROI':>8} {'#Contracts':>12}")
    print("-" * 70)
    for cat in sorted(cat_pnl.keys(), key=lambda x: cat_pnl[x]['pnl']):
        d = cat_pnl[cat]
        roi = d['pnl'] / d['cost'] * 100 if d['cost'] > 0 else 0
        print(f"{cat:<15} ${d['pnl']:>+9.2f} ${d['cost']:>9.2f} ${d['fees']:>7.2f} {roi:>+7.0f}% {d['trades']:>12}")

    # === THE KEY DIAGNOSIS ===
    print("\n" + "="*80)
    print("DIAGNOSIS: WHY THE GRADUAL DECLINES HAPPEN")
    print("="*80)
    print(f"""
The gradual decline between settlement jumps has THREE causes:

1. FEE DRAG: ${total_fees:.2f} total ({total_fees/400*100:.1f}% of deposit)
   - {taker_count} taker fills at avg ${total_taker_fees/max(taker_count,1):.4f}/fill
   - Every trade costs money even before you know if you win or lose

2. OVER-TRADING: {len(buy_fills)} buy fills deploying ${sum(f['cost'] for f in buy_fills):.0f} in capital
   - That's {sum(f['cost'] for f in buy_fills)/400:.1f}x the deposit turned over
   - More trades = more fees = faster decline

3. LOW WIN RATE ON CHEAP CONTRACTS:
   - 0-5c contracts: {price_win_rate['0-5c']['wins']}W/{price_win_rate['0-5c']['losses']}L = {price_win_rate['0-5c']['wins']/(max(1,price_win_rate['0-5c']['wins']+price_win_rate['0-5c']['losses']))*100:.0f}% win rate, P&L: ${price_win_rate['0-5c']['pnl']:+.2f}
   - 5-15c contracts: {price_win_rate['5-15c']['wins']}W/{price_win_rate['5-15c']['losses']}L = {price_win_rate['5-15c']['wins']/(max(1,price_win_rate['5-15c']['wins']+price_win_rate['5-15c']['losses']))*100:.0f}% win rate, P&L: ${price_win_rate['5-15c']['pnl']:+.2f}
   
   The bot buys hundreds of cheap lottery tickets that almost all expire worthless.
   Each one costs a fee + the contract price. The few that win don't cover the losses.

THE SIMPLEST FIX:
- STOP buying contracts below 30c (they have negative expected value)
- REDUCE trade frequency (fewer trades = less fee drag)
- ONLY trade when you have HIGH conviction (>60% edge)
- USE MAKER ORDERS to earn fee rebates instead of paying taker fees
""")

    # === WHAT MAKES THE JUMPS ===
    print("="*80)
    print("WHAT MAKES THE BIG JUMPS")
    print("="*80)
    print(f"""
The big upward jumps come from SETTLEMENT BATCHES where winning positions pay out.

The biggest winners were:
""")
    for ticker, data in sorted_winners[:5]:
        if data['pnl'] > 0:
            roi = data['pnl'] / data['cost'] * 100 if data['cost'] > 0 else 0
            print(f"  {ticker}: ${data['pnl']:+.2f} on ${data['cost']:.2f} invested ({roi:+.0f}% ROI)")

    print(f"""
Pattern: The big wins come from contracts bought at 30-70c that settle YES.
These are MODERATE probability events where the model had genuine edge.
NOT from cheap lottery tickets (0-5c) which have 0% win rate.

TO MAXIMIZE JUMPS:
- Focus on 30-70c contracts where you have genuine forecast edge
- Increase position size on high-conviction trades (fewer, bigger bets)
- Target markets where the model has demonstrated accuracy
""")

asyncio.run(main())
