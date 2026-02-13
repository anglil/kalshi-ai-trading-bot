#!/usr/bin/env python3
"""
Complete Dashboard Setup Script

This script:
1. Fixes database schema issues
2. Ensures all tables and columns exist
3. Tests database connectivity
4. Launches the dashboard

Run this to get your dashboard working properly.
"""

import asyncio
import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if dashboard requirements are installed."""
    required_packages = ['streamlit', 'pandas', 'plotly']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n📦 Installing missing packages...")
        
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing, check=True)
            print("✅ Packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run:")
            print(f"   pip install {' '.join(missing)}")
            return False
    
    return True

async def fix_database():
    """Fix database schema and ensure it's ready."""
    print("🔧 Fixing database schema...")
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.utils.database import DatabaseManager
    import aiosqlite
    
    db_manager = DatabaseManager()
    
    try:
        await db_manager.initialize()
        print("✅ Database initialized")
        
        # Check and fix positions table
        async with aiosqlite.connect(db_manager.db_path) as db:
            cursor = await db.execute("PRAGMA table_info(positions)")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            if 'strategy' not in column_names:
                print("🔧 Adding strategy column to positions table...")
                await db.execute("ALTER TABLE positions ADD COLUMN strategy TEXT")
                await db.commit()
                print("✅ Strategy column added to positions")
        
        # Check and fix trade_logs table
        async with aiosqlite.connect(db_manager.db_path) as db:
            cursor = await db.execute("PRAGMA table_info(trade_logs)")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            if 'strategy' not in column_names:
                print("🔧 Adding strategy column to trade_logs table...")
                await db.execute("ALTER TABLE trade_logs ADD COLUMN strategy TEXT")
                await db.commit()
                print("✅ Strategy column added to trade_logs")
        
        # Check and create llm_queries table
        async with aiosqlite.connect(db_manager.db_path) as db:
            cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='llm_queries'")
            table_exists = await cursor.fetchone()
            
            if not table_exists:
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
        performance = await db_manager.get_performance_by_strategy()
        print(f"✅ Performance query test passed: {len(performance)} strategies")
        
        # Test LLM query
        queries = await db_manager.get_llm_queries(hours_back=24, limit=1)
        print(f"✅ LLM query test passed: {len(queries)} queries")
        
        await db_manager.close()
        print("✅ Database ready for dashboard")
        return True
        
    except Exception as e:
        print(f"❌ Database fix failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("🚀 Launching dashboard...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "trading_dashboard.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Failed to launch dashboard: {e}")
        return False
    
    return True

async def main():
    """Main setup function."""
    print("🎯 Trading Dashboard Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("trading_dashboard.py").exists():
        print("❌ Error: trading_dashboard.py not found")
        print("💡 Make sure you're running this from the kalshi project root")
        return False
    
    # Check requirements
    print("📦 Checking requirements...")
    if not check_requirements():
        return False
    print("✅ Requirements satisfied")
    
    # Fix database
    print("\n🔧 Setting up database...")
    if not await fix_database():
        return False
    
    # Launch dashboard
    print("\n🌐 Dashboard will open at: http://localhost:8501")
    print("⏹️ Press Ctrl+C to stop")
    print("\n" + "="*50)
    
    launch_dashboard()
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    if not success:
        sys.exit(1) 