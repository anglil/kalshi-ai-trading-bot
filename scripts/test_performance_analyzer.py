#!/usr/bin/env python3

import asyncio
import sys
import json
sys.path.append('.')

from src.jobs.performance_analyzer import run_performance_analysis
from src.utils.logging_setup import get_trading_logger

async def test_performance_analyzer():
    """Test the automated performance analyzer with Grok4."""
    
    logger = get_trading_logger("test_analyzer")
    
    print("🚀 RUNNING AUTOMATED PERFORMANCE ANALYSIS WITH GROK4")
    print("=" * 60)
    
    try:
        # Run the comprehensive analysis
        report = await run_performance_analysis()
        
        if not report:
            print("❌ Analysis failed or returned empty report")
            return
        
        # Display results
        print("\n📊 ANALYSIS RESULTS")
        print("-" * 40)
        
        # Performance data summary
        perf_data = report.get('performance_data', {})
        portfolio = perf_data.get('portfolio', {})
        performance = perf_data.get('performance', {}).get('overall_stats', {})
        
        print(f"Portfolio Overview:")
        print(f"  Active Positions: {portfolio.get('active_positions', 0)}")
        print(f"  Total Contracts: {portfolio.get('total_contracts', 0)}")
        print(f"  Available Cash: ${portfolio.get('available_cash', 0):.2f}")
        
        print(f"\nHistorical Performance:")
        print(f"  Total Trades: {performance.get('total_trades', 0)}")
        print(f"  Win Rate: {performance.get('win_rate', 0):.1%}")
        print(f"  Total P&L: ${performance.get('total_pnl', 0):.2f}")
        print(f"  Average P&L: ${performance.get('avg_pnl', 0):.2f}")
        
        # AI Analysis
        ai_insights = report.get('ai_insights', {})
        if 'raw_analysis' in ai_insights:
            print(f"\n🤖 GROK4 ANALYSIS:")
            print("-" * 40)
            print(ai_insights['raw_analysis'])
        elif 'fallback_analysis' in ai_insights:
            print(f"\n📋 FALLBACK ANALYSIS:")
            print("-" * 40)
            print(ai_insights['fallback_analysis'])
        else:
            print("\n⚠️ No AI analysis available")
        
        # Action Items
        action_items = report.get('action_items', [])
        if action_items:
            print(f"\n🎯 ACTION ITEMS:")
            print("-" * 40)
            for i, item in enumerate(action_items, 1):
                priority = item.get('priority', 'Unknown')
                action = item.get('action', 'No action specified')
                rationale = item.get('rationale', 'No rationale provided')
                
                print(f"{i}. [{priority}] {action}")
                print(f"   Rationale: {rationale}")
                print()
        
        # Save detailed report
        with open('performance_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 Detailed report saved to: performance_analysis_report.json")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_performance_analyzer()) 