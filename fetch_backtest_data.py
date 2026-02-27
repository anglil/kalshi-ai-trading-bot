"""
Fetch historical public trade data from Kalshi for Follow the Leader backtesting.
We collect:
1. All public trades (taker_side, count, price, ticker, ts) via get_trades endpoint
2. Market metadata (close_time, open_time, result, series_ticker) for each market
3. Our own fills (to measure actual outcomes for markets we traded)
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timezone
from collections import defaultdict

sys.path.insert(0, '.')
from src.clients.kalshi_client import KalshiClient


async def fetch_all_trades(client, ticker, limit=1000):
    """Fetch all public trades for a specific market."""
    try:
        resp = await client._make_authenticated_request(
            'GET', f'/trade-api/v2/markets/{ticker}/trades',
            params={'limit': limit}
        )
        return resp.get('trades', [])
    except Exception as e:
        return []


async def fetch_market_details(client, ticker):
    """Fetch market metadata including result and close time."""
    try:
        resp = await client._make_authenticated_request(
            'GET', f'/trade-api/v2/markets/{ticker}'
        )
        return resp.get('market', {})
    except Exception as e:
        return {}


async def fetch_series_markets(client, series_ticker, limit=200):
    """Fetch all markets in a series."""
    try:
        resp = await client._make_authenticated_request(
            'GET', '/trade-api/v2/markets',
            params={'series_ticker': series_ticker, 'limit': limit, 'status': 'settled'}
        )
        return resp.get('markets', [])
    except Exception as e:
        return []


async def fetch_all_settled_markets(client, limit=200):
    """Fetch recently settled markets across all series."""
    try:
        resp = await client._make_authenticated_request(
            'GET', '/trade-api/v2/markets',
            params={'limit': limit, 'status': 'settled'}
        )
        return resp.get('markets', [])
    except Exception as e:
        return []


async def fetch_open_markets(client, limit=200):
    """Fetch currently open markets."""
    try:
        resp = await client._make_authenticated_request(
            'GET', '/trade-api/v2/markets',
            params={'limit': limit, 'status': 'open'}
        )
        return resp.get('markets', [])
    except Exception as e:
        return []


async def main():
    client = KalshiClient()

    print("Fetching own fills for outcome reference...")
    fills = await client.get_fills(limit=1000)
    print(f"  Got {len(fills)} fills")

    # Get unique tickers from our fills
    our_tickers = list(set(f['market_ticker'] for f in fills if 'market_ticker' in f))
    print(f"  Unique markets traded: {len(our_tickers)}")

    print("\nFetching settled markets for backtesting...")
    settled_markets = await fetch_all_settled_markets(client, limit=200)
    print(f"  Got {len(settled_markets)} settled markets")

    print("\nFetching open markets...")
    open_markets = await fetch_open_markets(client, limit=200)
    print(f"  Got {len(open_markets)} open markets")

    all_markets = settled_markets + open_markets
    print(f"\nTotal markets: {len(all_markets)}")

    # Filter to low-frequency markets (span >= 1 hour)
    low_freq_markets = []
    for m in all_markets:
        open_time = m.get('open_time', '')
        close_time = m.get('close_time', '') or m.get('expiration_time', '')
        if open_time and close_time:
            try:
                t_open = datetime.fromisoformat(open_time.replace('Z', '+00:00'))
                t_close = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
                span_hours = (t_close - t_open).total_seconds() / 3600
                if span_hours >= 1.0:
                    m['_span_hours'] = span_hours
                    low_freq_markets.append(m)
            except:
                pass

    print(f"Low-frequency markets (span >= 1h): {len(low_freq_markets)}")

    # Fetch public trades for each low-freq market
    print("\nFetching public trades for each market...")
    market_trades = {}
    for i, m in enumerate(low_freq_markets):
        ticker = m['ticker']
        trades = await fetch_all_trades(client, ticker, limit=1000)
        if trades:
            market_trades[ticker] = trades
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(low_freq_markets)} markets")

    print(f"\nMarkets with trade data: {len(market_trades)}")
    total_trades = sum(len(v) for v in market_trades.values())
    print(f"Total public trade records: {total_trades}")

    await client.close()

    # Save everything
    data = {
        'fills': fills,
        'our_tickers': our_tickers,
        'markets': {m['ticker']: m for m in all_markets},
        'low_freq_markets': [m['ticker'] for m in low_freq_markets],
        'market_trades': market_trades,
        'fetched_at': datetime.now(tz=timezone.utc).isoformat()
    }

    with open('/tmp/backtest_data.json', 'w') as f:
        json.dump(data, f)

    print(f"\nData saved to /tmp/backtest_data.json")
    print(f"File size: {os.path.getsize('/tmp/backtest_data.json') / 1024:.1f} KB")


if __name__ == '__main__':
    asyncio.run(main())
