#!/usr/bin/env python3

import asyncio
import sys
sys.path.append('.')

from src.jobs.automated_performance_analyzer import AutomatedPerformanceAnalyzer, run_performance_analysis

async def test_automated_analyzer():
    """Test the new automated performance analyzer."""
    
    print("🚀 TESTING AUTOMATED PERFORMANCE ANALYZER")
    print("=" * 60)
    
    try:
        # Run full analysis
        report = await run_performance_analysis()
        
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        # Display summary
        summary = report['summary']
        print(f"📊 SYSTEM HEALTH SCORE: {summary['overall_health_score']:.1f}/100")
        print(f"🔴 Critical Issues: {summary['total_critical_issues']}")
        print(f"⚠️  Warnings: {summary['total_warnings']}")
        print(f"🎯 Action Items: {len(report['action_items'])}")
        
        # Show performance metrics
        metrics = report['performance_metrics']
        print(f"\n📈 PERFORMANCE OVERVIEW:")
        print(f"   Total Trades: {metrics['total_trades']}")
        print(f"   Manual Trades: {metrics['manual_trades']} (Win Rate: {metrics['manual_win_rate']:.1%})")
        print(f"   Automated Trades: {metrics['automated_trades']} (Win Rate: {metrics['automated_win_rate']:.1%})")
        print(f"   Overall Win Rate: {metrics['overall_win_rate']:.1%}")
        print(f"   Total P&L: ${metrics['total_pnl']:.2f}")
        print(f"   Available Cash: ${metrics['available_cash']:.2f}")
        print(f"   Capital Utilization: {metrics['capital_utilization']:.1f}%")
        
        # Show critical risk checks
        critical_checks = [check for check in report['risk_checks'] if check['status'] == 'CRITICAL']
        if critical_checks:
            print(f"\n🔴 CRITICAL RISK ISSUES:")
            for check in critical_checks:
                print(f"   - {check['check_name']}: {check['recommendation']}")
        
        warning_checks = [check for check in report['risk_checks'] if check['status'] == 'WARNING']
        if warning_checks:
            print(f"\n⚠️  WARNING ISSUES:")
            for check in warning_checks:
                print(f"   - {check['check_name']}: {check['recommendation']}")
        
        # Show critical action items
        critical_actions = [action for action in report['action_items'] if action['priority'] == 'CRITICAL']
        if critical_actions:
            print(f"\n🚨 CRITICAL ACTIONS REQUIRED:")
            for action in critical_actions:
                print(f"   - {action['action']}")
                print(f"     Rationale: {action['rationale']}")
                print(f"     Target Date: {action['target_date']}")
                print(f"     Steps: {', '.join(action['implementation_steps'][:2])}...")
        
        high_priority_actions = [action for action in report['action_items'] if action['priority'] == 'HIGH']
        if high_priority_actions:
            print(f"\n🔶 HIGH PRIORITY ACTIONS:")
            for action in high_priority_actions:
                print(f"   - {action['action']}")
                print(f"     Target: {action['target_date']}")
        
        # Show Grok4 analysis if available
        if 'grok_analysis' in report and report['grok_analysis']['status'] == 'success':
            print(f"\n🤖 GROK4 AI ANALYSIS:")
            print("-" * 40)
            analysis_text = report['grok_analysis']['analysis_text']
            # Show first 500 characters
            print(analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text)
            print(f"\n💰 Grok4 Analysis Cost: ${report['grok_analysis']['cost']:.4f}")
        else:
            print(f"\n⚠️  Grok4 analysis unavailable, using fallback analysis")
        
        # Show total counts for each priority
        priority_counts = {}
        for action in report['action_items']:
            priority = action['priority']
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        print(f"\n📋 ACTION ITEMS BY PRIORITY:")
        for priority, count in priority_counts.items():
            print(f"   {priority}: {count} items")
        
        print(f"\n💾 Full report saved to database and JSON file")
        print("=" * 60)
        
        return report
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(test_automated_analyzer()) 