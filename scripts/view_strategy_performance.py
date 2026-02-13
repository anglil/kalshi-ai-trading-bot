#!/usr/bin/env python3
"""
Strategy Performance Viewer

View trading performance broken down by strategy to understand which approaches
are working best.
"""

import asyncio
import sys
import os
from datetime import datetime
import aiosqlite

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.database import DatabaseManager
from src.utils.logging_setup import setup_logging


async def view_strategy_performance():
    """Display performance metrics broken down by strategy."""
    
    setup_logging()
    
    print("📊 STRATEGY PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Initialize database
    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    try:
        # Get performance by strategy
        performance = await db_manager.get_performance_by_strategy()
        
        if not performance:
            print("ℹ️  No strategy performance data available yet.")
            print("   Strategies will appear here after trades are completed.")
            return
        
        # Display overall summary
        total_trades = sum(stats['completed_trades'] for stats in performance.values())
        total_pnl = sum(stats['total_pnl'] for stats in performance.values())
        total_open = sum(stats['open_positions'] for stats in performance.values())
        total_deployed = sum(stats['capital_deployed'] for stats in performance.values())
        
        print(f"📈 OVERALL PERFORMANCE:")
        print(f"   Total Completed Trades: {total_trades}")
        print(f"   Total P&L: ${total_pnl:.2f}")
        print(f"   Open Positions: {total_open}")
        print(f"   Capital Deployed: ${total_deployed:.2f}")
        print()
        
        # Display per-strategy breakdown
        print("🎯 PERFORMANCE BY STRATEGY:")
        print("-" * 60)
        
        # Sort strategies by total P&L (best first)
        sorted_strategies = sorted(
            performance.items(), 
            key=lambda x: x[1]['total_pnl'], 
            reverse=True
        )
        
        for strategy_name, stats in sorted_strategies:
            print(f"\n📋 {strategy_name.upper().replace('_', ' ')}")
            print(f"   Completed Trades: {stats['completed_trades']}")
            
            if stats['completed_trades'] > 0:
                print(f"   Total P&L: ${stats['total_pnl']:.2f}")
                print(f"   Avg P&L per Trade: ${stats['avg_pnl_per_trade']:.2f}")
                print(f"   Win Rate: {stats['win_rate_pct']:.1f}%")
                print(f"   Winning Trades: {stats['winning_trades']}")
                print(f"   Losing Trades: {stats['losing_trades']}")
                print(f"   Best Trade: ${stats['best_trade']:.2f}")
                print(f"   Worst Trade: ${stats['worst_trade']:.2f}")
            else:
                print(f"   Total P&L: No completed trades yet")
                
            if stats['open_positions'] > 0:
                print(f"   Open Positions: {stats['open_positions']}")
                print(f"   Capital Deployed: ${stats['capital_deployed']:.2f}")
        
        print("\n" + "=" * 60)
        
        # Strategy insights
        if total_trades > 0:
            best_strategy = max(performance.items(), key=lambda x: x[1]['total_pnl'])
            best_win_rate = max(
                [(k, v) for k, v in performance.items() if v['completed_trades'] > 0], 
                key=lambda x: x[1]['win_rate_pct'],
                default=(None, None)
            )
            
            print("💡 INSIGHTS:")
            print(f"   🏆 Best P&L: {best_strategy[0]} (${best_strategy[1]['total_pnl']:.2f})")
            
            if best_win_rate[0]:
                print(f"   🎯 Best Win Rate: {best_win_rate[0]} ({best_win_rate[1]['win_rate_pct']:.1f}%)")
            
            print(f"   💰 Average Trade: ${total_pnl/total_trades:.2f}")
        
    except Exception as e:
        print(f"❌ Error retrieving strategy performance: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await db_manager.close()


async def view_recent_positions_by_strategy():
    """View recent positions grouped by strategy."""
    
    print("\n📋 RECENT POSITIONS BY STRATEGY")
    print("=" * 60)
    
    # Initialize database
    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    try:
        # Get all positions grouped by strategy
        async with aiosqlite.connect(db_manager.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT 
                    strategy,
                    market_id, 
                    side, 
                    quantity, 
                    entry_price, 
                    status,
                    timestamp,
                    rationale
                FROM positions 
                WHERE strategy IS NOT NULL
                ORDER BY strategy, timestamp DESC
                LIMIT 50
            """)
            
            positions = await cursor.fetchall()
        
        if not positions:
            print("ℹ️  No positions with strategy information found.")
            return
        
        # Group by strategy
        by_strategy = {}
        for pos in positions:
            strategy = pos['strategy']
            if strategy not in by_strategy:
                by_strategy[strategy] = []
            by_strategy[strategy].append(pos)
        
        # Display each strategy's recent positions
        for strategy, pos_list in by_strategy.items():
            print(f"\n🎯 {strategy.upper().replace('_', ' ')} ({len(pos_list)} positions)")
            print("-" * 40)
            
            for pos in pos_list[:5]:  # Show top 5 recent
                timestamp = datetime.fromisoformat(pos['timestamp']).strftime("%m/%d %H:%M")
                print(f"   {timestamp} | {pos['market_id'][:20]}... | {pos['side']} {pos['quantity']} @ ${pos['entry_price']:.3f} | {pos['status']}")
            
            if len(pos_list) > 5:
                print(f"   ... and {len(pos_list) - 5} more positions")
    
    except Exception as e:
        print(f"❌ Error retrieving recent positions: {e}")
    
    finally:
        await db_manager.close()


if __name__ == "__main__":
    async def main():
        await view_strategy_performance()
        await view_recent_positions_by_strategy()
    
    asyncio.run(main()) 