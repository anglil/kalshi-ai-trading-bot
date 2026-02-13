#!/usr/bin/env python3
"""
Database initialization script for Kalshi AI Trading Bot
Creates the necessary database tables and schema
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from utils.database import DatabaseManager


async def init_database():
    """Initialize the database with required tables."""
    print("🗄️  Initializing database...")
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Create tables
        await db_manager.create_tables()
        
        print("✅ Database initialized successfully!")
        print("📊 Created tables:")
        print("   - markets")
        print("   - positions") 
        print("   - trade_logs")
        print("   - ai_analyses")
        print("   - performance_metrics")
        
        # Close database connection
        await db_manager.close()
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)


def main():
    """Main function."""
    print("🚀 Kalshi AI Trading Bot - Database Initialization")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  Warning: .env file not found")
        print("   Please create a .env file with your API keys before running the bot")
        print("   You can copy env.template to .env and fill in your keys")
    
    # Run database initialization
    asyncio.run(init_database())
    
    print("\n" + "=" * 50)
    print("🎉 Database setup completed!")
    print("\n📋 Next steps:")
    print("1. Make sure your .env file is configured with API keys")
    print("2. Activate your virtual environment")
    print("3. Run the bot: python beast_mode_bot.py")


if __name__ == "__main__":
    main() 