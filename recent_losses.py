"""
Analyze ALL recent losses — what settled, what's underwater, what the bot is doing RIGHT NOW.
"""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from src.clients.kalshi_client import KalshiClient

async def main():
    client = KalshiClient()
    
    # 1. Current balance
    balance = await client.get_balance()
    cash = balance.get('balance', 0)
    portfolio = balance.get('portfolio_value', 0)
    print(f"CURRENT STATE: ${cash/100:.2f} cash, ${portfolio/100:.2f} portfolio")
    print(f"Real P&L from $400 deposit: ${(portfolio - 40000)/100:.2f}")
    print()

    # 2. ALL settlements — see what settled recently and at what value
    all_settlements = []
    cursor = None
    for page in range(50):
        params = {'limit': 100}
        if cursor:
            params['cursor'] = cursor
        resp = await client._make_authenticated_request(
            'GET', '/trade-api/v2/portfolio/settlements', params=params
        )
        settlements = resp.get('settlements', [])
        all_settlements.extend(settlements)
        cursor = resp.get('cursor')
        if not settlements or not cursor:
            break
    
    print(f"=== ALL SETTLEMENTS ({len(all_settlements)} total) ===")
    print(f"{'Ticker':<50} {'Revenue':>8} {'Date':<25}")
    print("-" * 90)
    
    total_settle_rev = 0
    settle_wins = 0
    settle_losses = 0
    for s in sorted(all_settlements, key=lambda x: x.get('settled_time', x.get('created_time', '')), reverse=True):
        ticker = s.get('market_ticker', s.get('ticker', ''))
        revenue = s.get('revenue', 0)
        settled_time = s.get('settled_time', s.get('created_time', 'unknown'))
        result = s.get('market_result', s.get('result', ''))
        total_settle_rev += revenue
        if revenue > 0:
            settle_wins += 1
        else:
            settle_losses += 1
        print(f"  {ticker:<48} ${revenue/100:>7.2f}  {settled_time[:19]:<25} {result}")
    
    print(f"\nSettlement summary: {settle_wins} wins, {settle_losses} losses")
    print(f"Total settlement revenue: ${total_settle_rev/100:.2f}")

    # 3. ALL fills — get recent ones
    all_fills = []
    cursor = None
    for page in range(50):
        params = {'limit': 100}
        if cursor:
            params['cursor'] = cursor
        resp = await client._make_authenticated_request(
            'GET', '/trade-api/v2/portfolio/fills', params=params
        )
        fills = resp.get('fills', [])
        all_fills.extend(fills)
        cursor = resp.get('cursor')
        if not fills or not cursor:
            break
    
    # Group fills by ticker
    ticker_fills = defaultdict(lambda: {'buy_cost': 0, 'sell_rev': 0, 'contracts_bought': 0, 
                                         'contracts_sold': 0, 'sides': set(), 'last_fill': '',
                                         'taker_count': 0, 'total_count': 0})
    for f in all_fills:
        ticker = f.get('ticker', '')
        side = f.get('side', '')
        action = f.get('action', '')
        count = f.get('count', 0)
        price = f.get('yes_price', 0) if side == 'yes' else f.get('no_price', 0)
        cost = price * count
        is_taker = f.get('is_taker', False)
        created = f.get('created_time', '')
        
        td = ticker_fills[ticker]
        td['sides'].add(side)
        td['total_count'] += 1
        if is_taker:
            td['taker_count'] += 1
        if created > td['last_fill']:
            td['last_fill'] = created
        
        if action == 'buy':
            td['buy_cost'] += cost
            td['contracts_bought'] += count
        elif action == 'sell':
            td['sell_rev'] += cost
            td['contracts_sold'] += count

    # 4. Current positions with market prices
    positions_resp = await client._make_authenticated_request(
        'GET', '/trade-api/v2/portfolio/positions'
    )
    positions = positions_resp.get('market_positions', [])
    active = [p for p in positions if p.get('position', 0) != 0]
    
    print(f"\n=== CURRENT OPEN POSITIONS ({len(active)}) ===")
    print(f"{'Ticker':<50} {'Pos':>5} {'Exposure':>9} {'Side':<5} {'Fees':>6}")
    print("-" * 85)
    
    total_exposure = 0
    total_fees = 0
    position_details = []
    for p in sorted(active, key=lambda x: x.get('market_exposure', 0), reverse=True):
        ticker = p.get('ticker', '')
        pos = p.get('position', 0)
        exposure = p.get('market_exposure', 0)
        fees = p.get('total_traded', 0) - p.get('market_exposure', 0)  # approximate fees
        resting_orders = p.get('resting_orders_count', 0)
        
        # Get fill data for this ticker
        fd = ticker_fills.get(ticker, {})
        buy_cost = fd.get('buy_cost', 0)
        contracts = fd.get('contracts_bought', 0)
        avg_price = buy_cost / max(1, contracts)
        sides = fd.get('sides', set())
        
        total_exposure += exposure
        total_fees += max(0, fees)
        
        # Try to get current market price
        try:
            market_resp = await client._make_authenticated_request(
                'GET', f'/trade-api/v2/markets/{ticker}'
            )
            market = market_resp.get('market', {})
            yes_bid = market.get('yes_bid', 0)
            yes_ask = market.get('yes_ask', 0)
            no_bid = market.get('no_bid', 0)
            no_ask = market.get('no_ask', 0)
            close_time = market.get('close_time', '')
            status = market.get('status', '')
            
            # Determine our side and current value
            if pos > 0:
                side_str = 'YES'
                current_val = yes_bid * abs(pos)
                entry_price = avg_price
            else:
                side_str = 'NO'
                current_val = no_bid * abs(pos)
                entry_price = 100 - avg_price  # approximate
            
            unrealized_pnl = current_val - exposure
            
            position_details.append({
                'ticker': ticker, 'pos': pos, 'exposure': exposure,
                'current_val': current_val, 'unrealized_pnl': unrealized_pnl,
                'side': side_str, 'yes_bid': yes_bid, 'yes_ask': yes_ask,
                'close_time': close_time, 'status': status,
                'avg_entry': avg_price, 'contracts': contracts
            })
            
            pnl_str = f"${unrealized_pnl/100:>+7.2f}"
            price_str = f"bid={yes_bid}c ask={yes_ask}c" if pos > 0 else f"bid={no_bid}c ask={no_ask}c"
            print(f"  {ticker:<48} {pos:>5} ${exposure/100:>7.2f} {side_str:<5} {pnl_str} | {price_str} | close={close_time[:16]} {status}")
        except Exception as e:
            print(f"  {ticker:<48} {pos:>5} ${exposure/100:>7.2f} | error: {e}")
    
    print(f"\nTotal exposure: ${total_exposure/100:.2f}")
    
    # 5. Summarize underwater positions
    print(f"\n=== POSITION P&L SUMMARY ===")
    underwater = [p for p in position_details if p['unrealized_pnl'] < 0]
    profitable = [p for p in position_details if p['unrealized_pnl'] > 0]
    
    total_unrealized = sum(p['unrealized_pnl'] for p in position_details)
    print(f"Total unrealized P&L: ${total_unrealized/100:.2f}")
    print(f"Underwater positions: {len(underwater)}")
    print(f"Profitable positions: {len(profitable)}")
    
    if underwater:
        print(f"\n--- UNDERWATER (losing money) ---")
        for p in sorted(underwater, key=lambda x: x['unrealized_pnl']):
            print(f"  {p['ticker']:<48} {p['side']:>3} {abs(p['pos']):>4} contracts | "
                  f"cost=${p['exposure']/100:>6.2f} val=${p['current_val']/100:>6.2f} "
                  f"P&L=${p['unrealized_pnl']/100:>+7.2f} | {p['status']}")
    
    if profitable:
        print(f"\n--- PROFITABLE (making money) ---")
        for p in sorted(profitable, key=lambda x: x['unrealized_pnl'], reverse=True):
            print(f"  {p['ticker']:<48} {p['side']:>3} {abs(p['pos']):>4} contracts | "
                  f"cost=${p['exposure']/100:>6.2f} val=${p['current_val']/100:>6.2f} "
                  f"P&L=${p['unrealized_pnl']/100:>+7.2f} | {p['status']}")

    # 6. Check what the bot has been doing recently — last 24h fills
    print(f"\n=== RECENT FILLS (last 50) ===")
    recent = sorted(all_fills, key=lambda x: x.get('created_time', ''), reverse=True)[:50]
    for f in recent:
        ticker = f.get('ticker', '')
        action = f.get('action', '')
        side = f.get('side', '')
        count = f.get('count', 0)
        price = f.get('yes_price', 0) if side == 'yes' else f.get('no_price', 0)
        is_taker = f.get('is_taker', False)
        created = f.get('created_time', '')[:19]
        taker_str = "TAKER" if is_taker else "maker"
        print(f"  {created} {action:>4} {count:>4}x {side:>3} @ {price:>3}c  {taker_str:<6} {ticker}")

    # 7. Check pending/resting orders
    try:
        orders_resp = await client._make_authenticated_request(
            'GET', '/trade-api/v2/portfolio/orders', params={'status': 'resting', 'limit': 100}
        )
        orders = orders_resp.get('orders', [])
        print(f"\n=== RESTING ORDERS ({len(orders)}) ===")
        for o in orders:
            ticker = o.get('ticker', '')
            side = o.get('side', '')
            action = o.get('action', '')
            count = o.get('remaining_count', o.get('count', 0))
            price = o.get('yes_price', 0) if side == 'yes' else o.get('no_price', 0)
            print(f"  {action:>4} {count:>4}x {side:>3} @ {price:>3}c  {ticker}")
    except Exception as e:
        print(f"Orders error: {e}")

    await client.close()

asyncio.run(main())
