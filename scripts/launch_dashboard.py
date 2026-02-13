#!/usr/bin/env python3
"""
Trading Dashboard Launcher

Simple launcher for the comprehensive trading system dashboard.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit',
        'pandas', 
        'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print()
        print("📦 Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
        print()
        print("Or install all dashboard requirements:")
        print("   pip install -r dashboard_requirements.txt")
        return False
    
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    
    print("🚀 Trading System Dashboard Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    dashboard_path = Path("trading_dashboard.py")
    if not dashboard_path.exists():
        print("❌ Error: trading_dashboard.py not found in current directory")
        print("💡 Make sure you're running this from the kalshi project root")
        return False
    
    # Check requirements
    if not check_requirements():
        return False
    
    print("✅ All requirements satisfied")
    print("🌐 Launching dashboard...")
    print()
    print("📊 Dashboard will open in your browser at: http://localhost:8501")
    print("⏹️ Press Ctrl+C to stop the dashboard")
    print()
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, 
            "-m", 
            "streamlit", 
            "run", 
            "trading_dashboard.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = launch_dashboard()
    if not success:
        sys.exit(1) 