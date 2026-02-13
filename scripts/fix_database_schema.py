#!/usr/bin/env python3
"""
Database Schema Fix Script

This script manually runs database migrations and fixes any schema issues
to ensure the dashboard works properly.
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.database import DatabaseManager


async def fix_database_schema():
    """Fix database schema issues and run migrations."""
    
    print("🔧 Database Schema Fix Script")
    print("=" * 50)
    
    db_manager = DatabaseManager()
    
    try:
        print("📊 Initializing database and running migrations...")
        await db_manager.initialize()
        print("✅ Database initialization complete!")
        
        # Test strategy column in positions
        print("\n🔍 Checking positions table schema...")
        import aiosqlite
        async with aiosqlite.connect(db_manager.db_path) as db:
            cursor = await db.execute("PRAGMA table_info(positions)")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            if 'strategy' in column_names:
                print("✅ Strategy column exists in positions table")
            else:
                print("❌ Strategy column missing from positions table")
                print("🔧 Adding strategy column...")
                await db.execute("ALTER TABLE positions ADD COLUMN strategy TEXT")
                await db.commit()
                print("✅ Strategy column added to positions table")
        
        # Test strategy column in trade_logs
        print("\n🔍 Checking trade_logs table schema...")
        async with aiosqlite.connect(db_manager.db_path) as db:
            cursor = await db.execute("PRAGMA table_info(trade_logs)")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            if 'strategy' in column_names:
                print("✅ Strategy column exists in trade_logs table")
            else:
                print("❌ Strategy column missing from trade_logs table")
                print("🔧 Adding strategy column...")
                await db.execute("ALTER TABLE trade_logs ADD COLUMN strategy TEXT")
                await db.commit()
                print("✅ Strategy column added to trade_logs table")
        
        # Test llm_queries table
        print("\n🔍 Checking llm_queries table...")
        async with aiosqlite.connect(db_manager.db_path) as db:
            cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='llm_queries'")
            table_exists = await cursor.fetchone()
            
            if table_exists:
                print("✅ LLM queries table exists")
            else:
                print("❌ LLM queries table missing")
                print("🔧 Creating llm_queries table...")
                await db.execute("""
                    CREATE TABLE llm_queries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        query_type TEXT NOT NULL,
                        market_id TEXT,
                        prompt TEXT NOT NULL,
                        response TEXT NOT NULL,
                        tokens_used INTEGER,
                        cost_usd REAL,
                        confidence_extracted REAL,
                        decision_extracted TEXT
                    )
                """)
                await db.commit()
                print("✅ LLM queries table created")
        
        # Test performance query
        print("\n🔍 Testing performance query...")
        try:
            performance = await db_manager.get_performance_by_strategy()
            print(f"✅ Performance query successful: {len(performance)} strategies found")
            
            if performance:
                for strategy, stats in performance.items():
                    print(f"   - {strategy}: {stats['completed_trades']} trades, ${stats['total_pnl']:.2f} P&L")
            else:
                print("   ℹ️ No strategy performance data found (normal for new systems)")
                
        except Exception as e:
            print(f"❌ Performance query failed: {e}")
        
        # Test LLM query
        print("\n🔍 Testing LLM queries...")
        try:
            queries = await db_manager.get_llm_queries(hours_back=24, limit=5)
            print(f"✅ LLM query successful: {len(queries)} queries found")
            
            if queries:
                for query in queries:
                    print(f"   - {query.strategy}: {query.query_type} at {query.timestamp.strftime('%H:%M:%S')}")
            else:
                print("   ℹ️ No LLM queries found (normal until trading system runs)")
                
        except Exception as e:
            print(f"❌ LLM query failed: {e}")
        
        print("\n🎉 Database schema fix complete!")
        print("📊 Dashboard should now work properly")
        
    except Exception as e:
        print(f"❌ Error fixing database schema: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await db_manager.close()
    
    return True


async def verify_database_health():
    """Verify database health and show current state."""
    
    print("\n📋 Database Health Check")
    print("-" * 30)
    
    db_manager = DatabaseManager()
    
    try:
        await db_manager.initialize()
        
        # Count records in each table
        import aiosqlite
        async with aiosqlite.connect(db_manager.db_path) as db:
            # Markets
            cursor = await db.execute("SELECT COUNT(*) FROM markets")
            markets_count = (await cursor.fetchone())[0]
            
            # Positions
            cursor = await db.execute("SELECT COUNT(*) FROM positions")
            positions_count = (await cursor.fetchone())[0]
            
            # Trade logs
            cursor = await db.execute("SELECT COUNT(*) FROM trade_logs")
            trades_count = (await cursor.fetchone())[0]
            
            # LLM queries (if table exists)
            try:
                cursor = await db.execute("SELECT COUNT(*) FROM llm_queries")
                llm_count = (await cursor.fetchone())[0]
            except:
                llm_count = "Table not created yet"
        
        print(f"📊 Markets: {markets_count}")
        print(f"💼 Positions: {positions_count}")
        print(f"📈 Trades: {trades_count}")
        print(f"🤖 LLM Queries: {llm_count}")
        
    except Exception as e:
        print(f"❌ Database health check failed: {e}")
    
    finally:
        await db_manager.close()


if __name__ == "__main__":
    async def main():
        success = await fix_database_schema()
        await verify_database_health()
        
        if success:
            print("\n🚀 Ready to launch dashboard!")
            print("Run: python launch_dashboard.py")
        else:
            print("\n❌ Database fix failed - check errors above")
    
    asyncio.run(main()) 