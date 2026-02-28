"""Extract all trading data for audit."""
import asyncio, json, os, sqlite3
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()
from src.clients.kalshi_client import KalshiClient

async def main():
    client = KalshiClient()
    
    # 1. Balance
    balance = await client.get_balance()
    cash = balance.get('balance', 0) / 100
    portfolio = balance.get('portfolio_value', 0) / 100
    print(f"=== ACCOUNT ===")
    print(f"Cash: ${cash:.2f}")
    print(f"Portfolio value: ${portfolio:.2f}")
    print(f"Total: ${cash + portfolio:.2f}")
    
    # 2. Current positions from API
    positions = await client.get_positions()
    market_positions = positions.get('market_positions', [])
    print(f"\n=== LIVE POSITIONS FROM API ({len(market_positions)}) ===")
    total_exposure = 0
    for p in market_positions:
        ticker = p.get('ticker', '?')
        pos = p.get('position', 0)
        side = 'YES' if pos > 0 else 'NO'
        qty = abs(pos)
        exposure = p.get('market_exposure', 0)
        realized = p.get('realized_pnl', 0)
        resting = p.get('resting_orders_count', 0)
        total_exposure += exposure
        print(f"  {ticker}: {qty} {side} | exposure={exposure}c | realized_pnl={realized}c | resting={resting}")
    print(f"  TOTAL EXPOSURE: {total_exposure}c (${total_exposure/100:.2f})")
    
    # 3. Recent fills (trade history)
    fills = await client.get_fills(limit=100)
    fill_list = fills.get('fills', [])
    print(f"\n=== RECENT FILLS ({len(fill_list)}) ===")
    total_cost = 0
    total_fees = 0
    by_strategy = {}
    for f in fill_list:
        ticker = f.get('ticker', '?')
        side = f.get('side', '?')
        count = f.get('count', 0)
        price = f.get('yes_price', 0)
        no_price = f.get('no_price', 0)
        created = f.get('created_time', '')
        is_taker = f.get('is_taker', False)
        order_id = f.get('order_id', '')
        
        # Determine cost
        if side == 'yes':
            cost = price * count
        else:
            cost = no_price * count
        total_cost += cost
        
        print(f"  {created[:19]} | {ticker} | {count} {side.upper()} @ {price}c/{no_price}c | cost={cost}c | taker={is_taker}")
    
    # 4. Recent orders
    orders = await client.get_orders(status='resting')
    order_list = orders.get('orders', [])
    print(f"\n=== RESTING ORDERS ({len(order_list)}) ===")
    for o in order_list:
        ticker = o.get('ticker', '?')
        side = o.get('side', '?')
        price = o.get('yes_price', 0)
        remaining = o.get('remaining_count', 0)
        otype = o.get('type', '?')
        print(f"  {ticker}: {remaining} {side} @ {price}c ({otype})")
    
    # 5. Settled/closed orders
    settled = await client.get_orders(status='executed')
    settled_list = settled.get('orders', [])
    print(f"\n=== EXECUTED ORDERS ({len(settled_list)}) ===")
    
    # 6. Get trades (settlements)
    trades = await client.get_trades(limit=100)
    trade_list = trades.get('trades', []) if isinstance(trades, dict) else []
    print(f"\n=== RECENT TRADES ({len(trade_list)}) ===")
    for t in trade_list[:20]:
        print(f"  {json.dumps(t)}")
    
    # 7. DB positions and trade logs
    print(f"\n=== DATABASE POSITIONS ===")
    db = sqlite3.connect('trading_system.db')
    cursor = db.execute("SELECT market_id, side, quantity, entry_price, strategy, live, pnl, status, created_at FROM positions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    print(f"Total DB positions: {len(rows)}")
    
    strategy_pnl = {}
    strategy_count = {}
    strategy_wins = {}
    for r in rows:
        market_id, side, qty, entry_price, strategy, live, pnl, status, created = r
        pnl = pnl or 0
        if strategy not in strategy_pnl:
            strategy_pnl[strategy] = 0
            strategy_count[strategy] = 0
            strategy_wins[strategy] = 0
        strategy_pnl[strategy] += pnl
        strategy_count[strategy] += 1
        if pnl > 0:
            strategy_wins[strategy] += 1
        if live:
            print(f"  LIVE: {market_id} {side} x{qty} @ ${entry_price:.2f} | strategy={strategy} | pnl=${pnl:.2f}")
    
    print(f"\n=== P&L BY STRATEGY ===")
    for s in sorted(strategy_pnl.keys()):
        wins = strategy_wins[s]
        total = strategy_count[s]
        wr = wins/total*100 if total > 0 else 0
        print(f"  {s}: ${strategy_pnl[s]:.2f} P&L | {total} trades | {wins} wins ({wr:.0f}%)")
    
    # 8. Trade logs
    print(f"\n=== TRADE LOGS (last 50) ===")
    cursor = db.execute("SELECT * FROM trade_logs ORDER BY created_at DESC LIMIT 50")
    cols = [d[0] for d in cursor.description]
    for r in cursor.fetchall():
        row_dict = dict(zip(cols, r))
        print(f"  {row_dict.get('created_at','')} | {row_dict.get('market_id','')} | {row_dict.get('action','')} | {row_dict.get('side','')} | qty={row_dict.get('quantity','')} | price={row_dict.get('price','')} | pnl={row_dict.get('pnl','')} | strategy={row_dict.get('strategy','')}")
    
    # 9. Daily cost tracking
    print(f"\n=== DAILY COST TRACKING ===")
    try:
        cursor = db.execute("SELECT * FROM daily_cost_tracking ORDER BY date DESC LIMIT 10")
        cols = [d[0] for d in cursor.description]
        for r in cursor.fetchall():
            row_dict = dict(zip(cols, r))
            print(f"  {row_dict}")
    except:
        print("  No daily cost tracking table")
    
    db.close()
    await client.close()

asyncio.run(main())
