#!/usr/bin/env python3

import asyncio
import sys
sys.path.append('.')
from src.clients.kalshi_client import KalshiClient
from src.utils.database import DatabaseManager
import aiosqlite
from datetime import datetime, timedelta

async def analyze_performance():
    db = DatabaseManager()
    await db.initialize()
    
    # Get all trade logs
    async with aiosqlite.connect(db.db_path) as database:
        # Get all positions
        cursor = await database.execute("""
            SELECT market_id, side, quantity, entry_price, status, timestamp, live, rationale 
            FROM positions 
            ORDER BY timestamp DESC
        """)
        all_positions = await cursor.fetchall()
        
        # Get trade logs
        cursor = await database.execute("""
            SELECT * FROM trade_logs 
            ORDER BY timestamp DESC 
            LIMIT 20
        """)
        trade_logs = await cursor.fetchall()
        
    print("=== TRADING PERFORMANCE ANALYSIS ===")
    print(f"Total positions ever: {len(all_positions)}")
    
    open_positions = [p for p in all_positions if p[4] == 'open']
    closed_positions = [p for p in all_positions if p[4] == 'closed']
    
    print(f"Open positions: {len(open_positions)}")
    print(f"Closed positions: {len(closed_positions)}")
    
    # Analyze live vs non-live positions
    live_positions = [p for p in all_positions if p[6] == 1]  # live = 1
    non_live_positions = [p for p in all_positions if p[6] == 0]  # live = 0
    
    print(f"Live positions: {len(live_positions)}")
    print(f"Non-live positions: {len(non_live_positions)}")
    
    print("\n=== RECENT POSITIONS ===")
    for i, pos in enumerate(all_positions[:10]):
        market_id, side, quantity, entry_price, status, timestamp, live, rationale = pos
        print(f"{i+1}. {market_id[:30]}...")
        print(f"   {side} {quantity} @ ${entry_price:.2f} | Status: {status} | Live: {bool(live)}")
        print(f"   Time: {timestamp}")
        if rationale:
            print(f"   Rationale: {rationale[:100]}...")
        print()
    
    print("\n=== RECENT TRADE LOGS ===")
    for i, log in enumerate(trade_logs[:5]):
        print(f"{i+1}. {log}")
        print()

if __name__ == "__main__":
    asyncio.run(analyze_performance())