"""
Extract all trading data for deep analysis.
Uses KalshiClient's _make_authenticated_request for pagination.
"""
import asyncio
import json
import sys
from datetime import datetime, timezone
from urllib.parse import urlencode

sys.path.insert(0, '.')
from src.clients.kalshi_client import KalshiClient

async def fetch_all_fills(client):
    """Fetch all fills with pagination."""
    all_fills = []
    cursor = None
    page = 0
    while True:
        page += 1
        params = {'limit': 1000}
        if cursor:
            params['cursor'] = cursor
        resp = await client._make_authenticated_request('GET', '/trade-api/v2/portfolio/fills', params=params)
        fills = resp.get('fills', [])
        cursor = resp.get('cursor', None)
        all_fills.extend(fills)
        print(f"  Page {page}: {len(fills)} fills (total: {len(all_fills)})")
        if not fills or not cursor:
            break
    return all_fills

async def fetch_all_orders(client):
    """Fetch all orders with pagination."""
    all_orders = []
    cursor = None
    page = 0
    while True:
        page += 1
        params = {'limit': 1000}
        if cursor:
            params['cursor'] = cursor
        resp = await client._make_authenticated_request('GET', '/trade-api/v2/portfolio/orders', params=params)
        orders = resp.get('orders', [])
        cursor = resp.get('cursor', None)
        all_orders.extend(orders)
        print(f"  Page {page}: {len(orders)} orders (total: {len(all_orders)})")
        if not orders or not cursor:
            break
    return all_orders

async def fetch_all_positions(client):
    """Fetch all positions."""
    resp = await client._make_authenticated_request('GET', '/trade-api/v2/portfolio/positions', params={'limit': 1000})
    positions = resp.get('market_positions', resp.get('positions', []))
    
    # Also try settled
    settled = []
    try:
        resp2 = await client._make_authenticated_request('GET', '/trade-api/v2/portfolio/positions', 
                                                          params={'limit': 1000, 'settlement_status': 'settled'})
        settled = resp2.get('market_positions', resp2.get('positions', []))
    except Exception as e:
        print(f"  Could not fetch settled positions: {e}")
    
    return positions, settled

async def main():
    client = KalshiClient()
    data = {}
    
    print("Fetching balance...")
    data['balance'] = await client.get_balance()
    cash = data['balance'].get('balance', 0) / 100
    pv = data['balance'].get('portfolio_value', 0) / 100
    print(f"  Cash: ${cash:.2f}")
    print(f"  Portfolio Value: ${pv:.2f}")
    print(f"  Total: ${cash + pv:.2f}")
    
    print("\nFetching all fills...")
    data['fills'] = await fetch_all_fills(client)
    
    print("\nFetching all positions...")
    open_pos, settled_pos = await fetch_all_positions(client)
    data['open_positions'] = open_pos
    data['settled_positions'] = settled_pos
    print(f"  Open: {len(open_pos)}, Settled: {len(settled_pos)}")
    
    print("\nFetching all orders...")
    data['orders'] = await fetch_all_orders(client)
    
    # Try to get settlements
    print("\nFetching settlements...")
    try:
        resp = await client._make_authenticated_request('GET', '/trade-api/v2/portfolio/settlements', params={'limit': 1000})
        data['settlements'] = resp.get('settlements', [])
    except Exception as e:
        data['settlements'] = []
        print(f"  No settlements endpoint: {e}")
    print(f"  Settlements: {len(data['settlements'])}")
    
    await client.close()
    
    with open('/tmp/full_analysis_data.json', 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    # Quick summary
    print(f"\n{'='*60}")
    print(f"DATA EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Cash Balance:     ${cash:.2f}")
    print(f"Portfolio Value:  ${pv:.2f}")
    print(f"Total Account:    ${cash + pv:.2f}")
    print(f"Starting Deposit: $400.00")
    print(f"Total P&L:        ${cash + pv - 400:.2f}")
    print(f"Return:           {((cash + pv) / 400 - 1) * 100:.1f}%")
    print(f"Total Fills:      {len(data['fills'])}")
    print(f"Open Positions:   {len(data['open_positions'])}")
    print(f"Settled Pos:      {len(data['settled_positions'])}")
    print(f"Total Orders:     {len(data['orders'])}")
    print(f"{'='*60}")

asyncio.run(main())
