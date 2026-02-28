"""Analyze trade-level P&L from fills data and current positions."""
import asyncio, json, os
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()
from src.clients.kalshi_client import KalshiClient

async def main():
    client = KalshiClient()
    
    # Get all fills with pagination
    all_fills = []
    cursor = None
    for _ in range(20):  # max 20 pages
        params = {'limit': 100}
        if cursor:
            params['cursor'] = cursor
        resp = await client._make_authenticated_request('GET', '/trade-api/v2/portfolio/fills', params=params)
        fills = resp.get('fills', [])
        all_fills.extend(fills)
        cursor = resp.get('cursor', '')
        if not cursor or len(fills) < 100:
            break
    
    print(f"Total fills: {len(all_fills)}")
    
    # Get current positions
    resp = await client._make_authenticated_request('GET', '/trade-api/v2/portfolio/positions')
    positions = resp.get('market_positions', [])
    pos_map = {}
    for p in positions:
        pos_map[p['ticker']] = p
    
    # Aggregate fills by ticker
    ticker_data = defaultdict(lambda: {
        'yes_qty': 0, 'no_qty': 0, 
        'yes_cost': 0, 'no_cost': 0,
        'fills': [],
        'taker_count': 0, 'maker_count': 0,
        'taker_cost': 0, 'maker_cost': 0,
    })
    
    for f in all_fills:
        ticker = f.get('ticker', '?')
        side = f.get('side', '?')
        count = f.get('count', 0)
        yes_price = f.get('yes_price', 0)
        no_price = f.get('no_price', 0)
        is_taker = f.get('is_taker', False)
        created = f.get('created_time', '')
        
        d = ticker_data[ticker]
        if side == 'yes':
            d['yes_qty'] += count
            d['yes_cost'] += yes_price * count
        else:
            d['no_qty'] += count
            d['no_cost'] += no_price * count
        
        cost = (yes_price if side == 'yes' else no_price) * count
        if is_taker:
            d['taker_count'] += count
            d['taker_cost'] += cost
        else:
            d['maker_count'] += count
            d['maker_cost'] += cost
        d['fills'].append(f)
    
    # Categorize by market type
    weather_tickers = [t for t in ticker_data if 'HIGH' in t or 'LOW' in t or 'TEMP' in t]
    cpi_tickers = [t for t in ticker_data if 'CPI' in t]
    ftl_tickers = [t for t in ticker_data if 'CROSSCATEGORY' in t or 'SPORTSMULTI' in t or 'MVE' in t]
    other_tickers = [t for t in ticker_data if t not in weather_tickers + cpi_tickers + ftl_tickers]
    
    print(f"\n=== POSITION BREAKDOWN BY CATEGORY ===")
    categories = [
        ('WEATHER', weather_tickers),
        ('CPI/ECON', cpi_tickers),
        ('FTL (Follow the Leader)', ftl_tickers),
        ('OTHER', other_tickers),
    ]
    
    grand_total_cost = 0
    grand_total_exposure = 0
    
    for cat_name, tickers in categories:
        cat_cost = 0
        cat_exposure = 0
        cat_realized = 0
        cat_taker_cost = 0
        cat_maker_cost = 0
        cat_positions = 0
        
        print(f"\n--- {cat_name} ({len(tickers)} tickers) ---")
        for t in sorted(tickers):
            d = ticker_data[t]
            total_cost = d['yes_cost'] + d['no_cost']
            cat_cost += total_cost
            cat_taker_cost += d['taker_cost']
            cat_maker_cost += d['maker_cost']
            
            # Current position from API
            pos = pos_map.get(t, {})
            exposure = pos.get('market_exposure', 0)
            realized = pos.get('realized_pnl', 0)
            current_pos = pos.get('position', 0)
            cat_exposure += exposure
            cat_realized += realized
            if current_pos != 0:
                cat_positions += 1
            
            # Check for contradictory fills (both YES and NO)
            contradiction = ""
            if d['yes_qty'] > 0 and d['no_qty'] > 0:
                contradiction = " *** CONTRADICTION: bought both YES and NO ***"
            
            print(f"  {t}: bought {d['yes_qty']} YES (${d['yes_cost']/100:.2f}) + {d['no_qty']} NO (${d['no_cost']/100:.2f}) | total=${total_cost/100:.2f} | exposure=${exposure/100:.2f} | realized=${realized/100:.2f} | taker%={d['taker_count']/(d['taker_count']+d['maker_count'])*100:.0f}%{contradiction}")
        
        grand_total_cost += cat_cost
        grand_total_exposure += cat_exposure
        taker_pct = cat_taker_cost / (cat_taker_cost + cat_maker_cost) * 100 if (cat_taker_cost + cat_maker_cost) > 0 else 0
        
        print(f"  SUBTOTAL: cost=${cat_cost/100:.2f} | exposure=${cat_exposure/100:.2f} | realized=${cat_realized/100:.2f} | open_positions={cat_positions} | taker%={taker_pct:.0f}%")
    
    print(f"\n=== GRAND TOTAL ===")
    print(f"Total capital deployed: ${grand_total_cost/100:.2f}")
    print(f"Current exposure: ${grand_total_exposure/100:.2f}")
    print(f"Cash: $0.06")
    print(f"Portfolio value: $274.32")
    
    # Estimate total P&L
    # Capital deployed - (current portfolio value) = approximate loss
    print(f"\n=== ESTIMATED LOSS ANALYSIS ===")
    print(f"If started with ~$1000:")
    print(f"  Current value: $274.38")
    print(f"  Estimated loss: ~$725.62")
    
    # Check for settled positions (realized_pnl != 0)
    print(f"\n=== POSITIONS WITH REALIZED P&L ===")
    for t in sorted(pos_map.keys()):
        p = pos_map[t]
        rpnl = p.get('realized_pnl', 0)
        if rpnl != 0:
            pos = p.get('position', 0)
            side = 'YES' if pos > 0 else 'NO' if pos < 0 else 'CLOSED'
            print(f"  {t}: {abs(pos)} {side} | realized_pnl=${rpnl/100:.2f}")
    
    await client.close()

asyncio.run(main())
