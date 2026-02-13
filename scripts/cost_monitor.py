#!/usr/bin/env python3
"""
Cost Monitor - Real-time AI spending tracker for Kalshi Trading System

Usage:
    python cost_monitor.py           # Show today's costs
    python cost_monitor.py --week    # Show weekly analysis
    python cost_monitor.py --live    # Live monitoring mode
"""

import asyncio
import argparse
from datetime import datetime, timedelta
from src.utils.database import DatabaseManager
from src.config.settings import settings

async def get_cost_summary(db_manager: DatabaseManager, days: int = 1) -> dict:
    """Get cost summary for the specified number of days."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    import aiosqlite
    async with aiosqlite.connect(db_manager.db_path) as db:
        # Daily costs breakdown
        cursor = await db.execute("""
            SELECT date, total_ai_cost, analysis_count, decision_count
            FROM daily_cost_tracking 
            WHERE date >= ?
            ORDER BY date DESC
        """, (start_date.strftime('%Y-%m-%d'),))
        daily_data = await cursor.fetchall()
        
        # Total summary
        cursor = await db.execute("""
            SELECT 
                SUM(total_ai_cost) as total_cost,
                SUM(analysis_count) as total_analyses,
                SUM(decision_count) as total_decisions
            FROM daily_cost_tracking 
            WHERE date >= ?
        """, (start_date.strftime('%Y-%m-%d'),))
        totals = await cursor.fetchone()
        
        # Most analyzed markets
        cursor = await db.execute("""
            SELECT market_id, COUNT(*) as count, SUM(cost_usd) as total_cost
            FROM market_analyses 
            WHERE DATE(analysis_timestamp) >= ?
            GROUP BY market_id
            ORDER BY total_cost DESC
            LIMIT 5
        """, (start_date.strftime('%Y-%m-%d'),))
        top_markets = await cursor.fetchall()
        
        return {
            'daily_data': daily_data,
            'totals': totals,
            'top_markets': top_markets,
            'budget_limit': settings.trading.daily_ai_budget
        }

def print_cost_report(summary: dict, days: int):
    """Print formatted cost report."""
    print(f"\n💰 AI Cost Report - Last {days} day(s)")
    print("=" * 50)
    
    if summary['totals'] and summary['totals'][0]:
        total_cost = summary['totals'][0]
        total_analyses = summary['totals'][1] or 0
        total_decisions = summary['totals'][2] or 0
        
        print(f"📊 Summary:")
        print(f"   Total Cost: ${total_cost:.3f}")
        print(f"   Total Analyses: {total_analyses}")
        print(f"   Total Decisions: {total_decisions}")
        print(f"   Cost per Analysis: ${total_cost/max(1, total_analyses):.4f}")
        print(f"   Analysis→Decision Rate: {total_decisions/max(1, total_analyses):.1%}")
        
        # Budget status for today
        if days == 1:
            today_data = next((d for d in summary['daily_data'] if d[0] == datetime.now().strftime('%Y-%m-%d')), None)
            if today_data:
                today_cost = today_data[1]
                budget_used = today_cost / summary['budget_limit']
                status = "🟢" if budget_used < 0.5 else "🟡" if budget_used < 0.8 else "🔴"
                print(f"\n{status} Today's Budget: ${today_cost:.3f} / ${summary['budget_limit']} ({budget_used:.1%})")
    else:
        print("📊 No cost data available for this period")
    
    # Daily breakdown
    if summary['daily_data']:
        print(f"\n📅 Daily Breakdown:")
        for date, cost, analyses, decisions in summary['daily_data'][:7]:  # Show max 7 days
            rate = f"{decisions/max(1, analyses):.1%}" if analyses > 0 else "0%"
            print(f"   {date}: ${cost:.3f} ({analyses} analyses → {decisions} decisions, {rate})")
    
    # Top markets by cost
    if summary['top_markets']:
        print(f"\n🎯 Most Expensive Markets:")
        for market_id, count, cost in summary['top_markets']:
            print(f"   {market_id[:40]:<40} ${cost:.3f} ({count} analyses)")
    
    print("\n" + "=" * 50)

async def live_monitor(db_manager: DatabaseManager):
    """Live monitoring mode - updates every 30 seconds."""
    print("🔴 LIVE MONITORING MODE (Ctrl+C to exit)")
    print("Updates every 30 seconds...\n")
    
    try:
        while True:
            summary = await get_cost_summary(db_manager, days=1)
            
            # Clear screen and show current status
            print("\033[2J\033[H")  # Clear screen
            print_cost_report(summary, 1)
            
            # Show current time
            print(f"🕐 Last updated: {datetime.now().strftime('%H:%M:%S')}")
            print("   Press Ctrl+C to exit")
            
            await asyncio.sleep(30)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

async def main():
    parser = argparse.ArgumentParser(description="Monitor AI costs for Kalshi Trading System")
    parser.add_argument("--week", action="store_true", help="Show weekly summary")
    parser.add_argument("--live", action="store_true", help="Live monitoring mode")
    args = parser.parse_args()
    
    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    if args.live:
        await live_monitor(db_manager)
    elif args.week:
        summary = await get_cost_summary(db_manager, days=7)
        print_cost_report(summary, 7)
    else:
        summary = await get_cost_summary(db_manager, days=1)
        print_cost_report(summary, 1)

if __name__ == "__main__":
    # Fix import
    import aiosqlite
    asyncio.run(main()) 