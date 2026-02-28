"""
Deep analysis of all trading activity to identify the dominant loss driver.
Pulls fills, settlements, orders, and positions from Kalshi API.
"""
import asyncio
import json
from datetime import datetime, timedelta
from collections import defaultdict
from src.clients.kalshi_client import KalshiClient

async def main():
    client = KalshiClient()
    out = {}

    # 1. Balance
    balance = await client.get_balance()
    out['balance'] = balance
    print(f"Balance: ${balance.get('balance', 0)/100:.2f} cash, ${balance.get('portfolio_value', 0)/100:.2f} portfolio")

    # 2. Get ALL fills (paginated)
    all_fills = []
    cursor = None
    for page in range(50):
        params = {'limit': 100}
        if cursor:
            params['cursor'] = cursor
        try:
            resp = await client._make_authenticated_request(
                'GET', '/trade-api/v2/portfolio/fills', params=params
            )
            fills = resp.get('fills', [])
            all_fills.extend(fills)
            cursor = resp.get('cursor')
            if not fills or not cursor:
                break
        except Exception as e:
            print(f"Fills page {page} error: {e}")
            break
    
    print(f"\nTotal fills: {len(all_fills)}")

    # 3. Get ALL settlements (paginated)
    all_settlements = []
    cursor = None
    for page in range(50):
        params = {'limit': 100}
        if cursor:
            params['cursor'] = cursor
        try:
            resp = await client._make_authenticated_request(
                'GET', '/trade-api/v2/portfolio/settlements', params=params
            )
            settlements = resp.get('settlements', [])
            all_settlements.extend(settlements)
            cursor = resp.get('cursor')
            if not settlements or not cursor:
                break
        except Exception as e:
            print(f"Settlements page {page} error: {e}")
            break
    
    print(f"Total settlements: {len(all_settlements)}")

    # 4. Current positions
    positions_resp = await client._make_authenticated_request(
        'GET', '/trade-api/v2/portfolio/positions'
    )
    positions = [mp for mp in positions_resp.get('market_positions', []) if mp.get('position', 0) != 0]
    print(f"Active positions: {len(positions)}")

    # 5. Classify tickers
    def classify(ticker):
        t = ticker.upper()
        if any(k in t for k in ['HIGH', 'LOW', 'TEMP', 'RAIN', 'SNOW']):
            return 'weather'
        elif 'CPI' in t:
            return 'economics'
        elif 'NBA' in t:
            return 'sports'
        elif any(k in t for k in ['GREENLAND', 'KHAMENEI', 'MEDIA']):
            return 'political'
        elif 'MV' in t or 'ESPORT' in t:
            return 'esports'
        else:
            return 'other'

    # 6. Analyze fills by ticker — compute cost basis
    ticker_data = defaultdict(lambda: {
        'buys': [], 'sells': [], 'buy_cost_cents': 0, 'sell_revenue_cents': 0,
        'contracts_bought': 0, 'contracts_sold': 0, 'fees_cents': 0,
        'taker_fills': 0, 'maker_fills': 0,
    })
    
    for f in all_fills:
        ticker = f.get('ticker', '')
        side = f.get('side', '')
        action = f.get('action', '')
        count = f.get('count', 0)
        yes_price = f.get('yes_price', 0)
        no_price = f.get('no_price', 0)
        is_taker = f.get('is_taker', False)
        
        price = yes_price if side == 'yes' else no_price
        cost_cents = price * count
        
        td = ticker_data[ticker]
        if action == 'buy':
            td['buys'].append(f)
            td['buy_cost_cents'] += cost_cents
            td['contracts_bought'] += count
        elif action == 'sell':
            td['sells'].append(f)
            td['sell_revenue_cents'] += cost_cents
            td['contracts_sold'] += count
        
        if is_taker:
            td['taker_fills'] += 1
        else:
            td['maker_fills'] += 1

    # 7. Map settlements by ticker
    settlement_by_ticker = defaultdict(lambda: {'revenue_cents': 0, 'count': 0})
    for s in all_settlements:
        ticker = s.get('market_ticker', s.get('ticker', ''))
        revenue = s.get('revenue', 0)
        settlement_by_ticker[ticker]['revenue_cents'] += revenue
        settlement_by_ticker[ticker]['count'] += 1

    # 8. Compute per-ticker P&L
    ticker_pnl = {}
    for ticker, td in ticker_data.items():
        settlement_rev = settlement_by_ticker.get(ticker, {}).get('revenue_cents', 0)
        net_pnl = td['sell_revenue_cents'] + settlement_rev - td['buy_cost_cents']
        
        cat = classify(ticker)
        settled = ticker in settlement_by_ticker
        
        ticker_pnl[ticker] = {
            'category': cat,
            'buy_cost_cents': td['buy_cost_cents'],
            'sell_revenue_cents': td['sell_revenue_cents'],
            'settlement_revenue_cents': settlement_rev,
            'net_pnl_cents': net_pnl,
            'contracts_bought': td['contracts_bought'],
            'contracts_sold': td['contracts_sold'],
            'avg_buy_price': td['buy_cost_cents'] / max(1, td['contracts_bought']),
            'settled': settled,
            'taker_fills': td['taker_fills'],
            'maker_fills': td['maker_fills'],
            'total_fills': td['taker_fills'] + td['maker_fills'],
        }

    # 9. Category aggregation
    cat_agg = defaultdict(lambda: {
        'buy_cost': 0, 'sell_rev': 0, 'settle_rev': 0, 'net_pnl': 0,
        'tickers': 0, 'contracts': 0, 'settled_wins': 0, 'settled_losses': 0,
        'taker_fills': 0, 'maker_fills': 0,
    })
    
    for ticker, d in ticker_pnl.items():
        cat = d['category']
        cat_agg[cat]['buy_cost'] += d['buy_cost_cents']
        cat_agg[cat]['sell_rev'] += d['sell_revenue_cents']
        cat_agg[cat]['settle_rev'] += d['settlement_revenue_cents']
        cat_agg[cat]['net_pnl'] += d['net_pnl_cents']
        cat_agg[cat]['tickers'] += 1
        cat_agg[cat]['contracts'] += d['contracts_bought']
        cat_agg[cat]['taker_fills'] += d['taker_fills']
        cat_agg[cat]['maker_fills'] += d['maker_fills']
        if d['settled']:
            if d['settlement_revenue_cents'] > 0:
                cat_agg[cat]['settled_wins'] += 1
            elif d['settlement_revenue_cents'] == 0 and d['buy_cost_cents'] > 0:
                cat_agg[cat]['settled_losses'] += 1
            else:
                cat_agg[cat]['settled_losses'] += 1

    # 10. Print category breakdown
    print("\n" + "="*90)
    print("CATEGORY P&L BREAKDOWN")
    print("="*90)
    print(f"{'Category':<12} {'Buy Cost':>10} {'Sell Rev':>10} {'Settle Rev':>10} {'Net P&L':>10} {'Tickers':>8} {'Contracts':>10} {'Win Rate':>9} {'Taker%':>7}")
    print("-"*90)
    
    total_buy = 0
    total_sell = 0
    total_settle = 0
    total_net = 0
    
    for cat in sorted(cat_agg.keys(), key=lambda x: cat_agg[x]['net_pnl']):
        d = cat_agg[cat]
        total_settled = d['settled_wins'] + d['settled_losses']
        wr = d['settled_wins'] / max(1, total_settled) * 100
        total_fills = d['taker_fills'] + d['maker_fills']
        taker_pct = d['taker_fills'] / max(1, total_fills) * 100
        
        print(f"{cat:<12} ${d['buy_cost']/100:>9.2f} ${d['sell_rev']/100:>9.2f} ${d['settle_rev']/100:>9.2f} ${d['net_pnl']/100:>9.2f} {d['tickers']:>8} {d['contracts']:>10} {wr:>7.0f}% {taker_pct:>6.0f}%")
        
        total_buy += d['buy_cost']
        total_sell += d['sell_rev']
        total_settle += d['settle_rev']
        total_net += d['net_pnl']
    
    print("-"*90)
    print(f"{'TOTAL':<12} ${total_buy/100:>9.2f} ${total_sell/100:>9.2f} ${total_settle/100:>9.2f} ${total_net/100:>9.2f}")

    # 11. Top 20 losing tickers
    print("\n" + "="*90)
    print("TOP 20 LOSING TICKERS")
    print("="*90)
    sorted_t = sorted(ticker_pnl.items(), key=lambda x: x[1]['net_pnl_cents'])
    for ticker, d in sorted_t[:20]:
        print(f"  {ticker:<45} P&L=${d['net_pnl_cents']/100:>8.2f} | "
              f"bought {d['contracts_bought']:>3} @ avg {d['avg_buy_price']:>4.0f}c | "
              f"settle=${d['settlement_revenue_cents']/100:>6.2f} | "
              f"taker={d['taker_fills']}/{d['total_fills']} | {d['category']}")

    # 12. Top 10 winning tickers
    print("\n" + "="*90)
    print("TOP 10 WINNING TICKERS")
    print("="*90)
    for ticker, d in sorted_t[-10:]:
        print(f"  {ticker:<45} P&L=${d['net_pnl_cents']/100:>8.2f} | "
              f"bought {d['contracts_bought']:>3} @ avg {d['avg_buy_price']:>4.0f}c | "
              f"settle=${d['settlement_revenue_cents']/100:>6.2f} | {d['category']}")

    # 13. Weather-specific deep dive
    print("\n" + "="*90)
    print("WEATHER DEEP DIVE")
    print("="*90)
    
    weather_tickers = {t: d for t, d in ticker_pnl.items() if d['category'] == 'weather'}
    settled_weather = {t: d for t, d in weather_tickers.items() if d['settled']}
    unsettled_weather = {t: d for t, d in weather_tickers.items() if not d['settled']}
    
    # Parse city from ticker
    def get_city(ticker):
        t = ticker.upper()
        if 'CHI' in t: return 'Chicago'
        if 'MIA' in t: return 'Miami'
        if 'AUS' in t: return 'Austin'
        if 'NY' in t: return 'New York'
        if 'LA' in t or 'LAX' in t: return 'Los Angeles'
        if 'DEN' in t: return 'Denver'
        if 'ATL' in t: return 'Atlanta'
        return 'Unknown'
    
    # Parse date from ticker
    def get_date(ticker):
        import re
        m = re.search(r'26([A-Z]{3})(\d{2})', ticker)
        if m:
            return f"Feb {m.group(2)}" if m.group(1) == 'FEB' else f"{m.group(1)} {m.group(2)}"
        return 'Unknown'
    
    # City-level P&L
    city_pnl = defaultdict(lambda: {'pnl': 0, 'cost': 0, 'contracts': 0, 'tickers': 0, 'wins': 0, 'losses': 0})
    for t, d in weather_tickers.items():
        city = get_city(t)
        city_pnl[city]['pnl'] += d['net_pnl_cents']
        city_pnl[city]['cost'] += d['buy_cost_cents']
        city_pnl[city]['contracts'] += d['contracts_bought']
        city_pnl[city]['tickers'] += 1
        if d['settled'] and d['settlement_revenue_cents'] > 0:
            city_pnl[city]['wins'] += 1
        elif d['settled']:
            city_pnl[city]['losses'] += 1
    
    print("\n--- Weather P&L by City ---")
    print(f"{'City':<15} {'P&L':>10} {'Cost':>10} {'Contracts':>10} {'Tickers':>8} {'W/L':>8}")
    print("-"*65)
    for city in sorted(city_pnl.keys(), key=lambda x: city_pnl[x]['pnl']):
        d = city_pnl[city]
        print(f"{city:<15} ${d['pnl']/100:>9.2f} ${d['cost']/100:>9.2f} {d['contracts']:>10} {d['tickers']:>8} {d['wins']}W/{d['losses']}L")

    # Settled weather detail
    print(f"\n--- Settled Weather ({len(settled_weather)} tickers) ---")
    wins = sum(1 for d in settled_weather.values() if d['settlement_revenue_cents'] > 0)
    losses = len(settled_weather) - wins
    total_settle_rev = sum(d['settlement_revenue_cents'] for d in settled_weather.values())
    total_buy_cost = sum(d['buy_cost_cents'] for d in settled_weather.values())
    print(f"  Win rate: {wins}/{len(settled_weather)} = {wins/max(1,len(settled_weather))*100:.0f}%")
    print(f"  Total buy cost: ${total_buy_cost/100:.2f}")
    print(f"  Total settlement revenue: ${total_settle_rev/100:.2f}")
    print(f"  Net settled P&L: ${(total_settle_rev - total_buy_cost)/100:.2f}")
    
    # Show each settled weather ticker
    for t in sorted(settled_weather.keys(), key=lambda x: settled_weather[x]['net_pnl_cents']):
        d = settled_weather[t]
        result = "WIN" if d['settlement_revenue_cents'] > 0 else "LOSS"
        print(f"    {t:<45} {result:>4} | cost=${d['buy_cost_cents']/100:>6.2f} settle=${d['settlement_revenue_cents']/100:>6.2f} P&L=${d['net_pnl_cents']/100:>7.2f} | {d['contracts_bought']} contracts")

    # Unsettled weather
    print(f"\n--- Unsettled Weather ({len(unsettled_weather)} tickers, current risk) ---")
    total_unsettled_cost = sum(d['buy_cost_cents'] for d in unsettled_weather.values())
    print(f"  Total cost at risk: ${total_unsettled_cost/100:.2f}")
    for t in sorted(unsettled_weather.keys(), key=lambda x: unsettled_weather[x]['buy_cost_cents'], reverse=True):
        d = unsettled_weather[t]
        print(f"    {t:<45} cost=${d['buy_cost_cents']/100:>6.2f} | {d['contracts_bought']} contracts @ avg {d['avg_buy_price']:.0f}c")

    # 14. Taker vs Maker analysis
    print("\n" + "="*90)
    print("TAKER vs MAKER ANALYSIS")
    print("="*90)
    total_taker = sum(1 for f in all_fills if f.get('is_taker', False))
    total_maker = len(all_fills) - total_taker
    print(f"Taker fills: {total_taker} ({total_taker/max(1,len(all_fills))*100:.0f}%)")
    print(f"Maker fills: {total_maker} ({total_maker/max(1,len(all_fills))*100:.0f}%)")
    
    # Taker cost analysis
    taker_cost = sum(f.get('count', 0) * (f.get('yes_price', 0) if f.get('side') == 'yes' else f.get('no_price', 0)) 
                     for f in all_fills if f.get('is_taker', False) and f.get('action') == 'buy')
    maker_cost = sum(f.get('count', 0) * (f.get('yes_price', 0) if f.get('side') == 'yes' else f.get('no_price', 0))
                     for f in all_fills if not f.get('is_taker', False) and f.get('action') == 'buy')
    print(f"Taker buy volume: ${taker_cost/100:.2f}")
    print(f"Maker buy volume: ${maker_cost/100:.2f}")

    # 15. Position size distribution for weather
    print("\n" + "="*90)
    print("WEATHER POSITION SIZE ANALYSIS")
    print("="*90)
    weather_sizes = [(t, d['contracts_bought'], d['buy_cost_cents']/100) 
                     for t, d in weather_tickers.items()]
    weather_sizes.sort(key=lambda x: x[1], reverse=True)
    print(f"{'Ticker':<45} {'Contracts':>10} {'Cost':>10}")
    print("-"*70)
    for t, contracts, cost in weather_sizes[:20]:
        print(f"{t:<45} {contracts:>10} ${cost:>9.2f}")
    
    total_weather_contracts = sum(x[1] for x in weather_sizes)
    total_weather_cost = sum(x[2] for x in weather_sizes)
    print(f"\nTotal weather contracts: {total_weather_contracts}")
    print(f"Total weather cost: ${total_weather_cost:.2f}")
    print(f"Avg cost per contract: ${total_weather_cost/max(1,total_weather_contracts):.2f}")

    # Save raw data
    with open('/home/ubuntu/deep_analysis_raw.json', 'w') as f:
        json.dump({
            'fills_count': len(all_fills),
            'settlements_count': len(all_settlements),
            'category_pnl': {k: dict(v) for k, v in cat_agg.items()},
            'ticker_pnl': ticker_pnl,
            'total_buy_cents': total_buy,
            'total_sell_cents': total_sell,
            'total_settle_cents': total_settle,
            'total_net_cents': total_net,
        }, f, indent=2, default=str)
    
    print("\n\nRaw data saved to /home/ubuntu/deep_analysis_raw.json")
    await client.close()

asyncio.run(main())
