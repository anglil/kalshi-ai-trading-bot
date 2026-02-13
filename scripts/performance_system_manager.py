#!/usr/bin/env python3

"""
Performance System Manager

Comprehensive orchestration system for the automated Kalshi trading performance analyzer.
This is the main entry point for managing the entire performance analysis ecosystem.

Features:
- Start/stop automated scheduler
- Run on-demand analysis
- Emergency intervention tools
- Dashboard integration
- System health monitoring
- Grok4-powered insights
- Risk management automation
"""

import asyncio
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional
import signal

# Add current directory to path
sys.path.append('.')

from src.jobs.automated_performance_analyzer import run_performance_analysis
from src.jobs.performance_scheduler import (
    start_performance_scheduler, 
    stop_performance_scheduler, 
    get_scheduler_status,
    ScheduleConfig
)
from src.jobs.performance_dashboard_integration import (
    dashboard_get_summary,
    dashboard_trigger_analysis,
    dashboard_get_metrics,
    dashboard_get_alerts
)
from src.utils.logging_setup import setup_logging, get_trading_logger


class PerformanceSystemManager:
    """
    Main orchestrator for the automated performance analysis system.
    
    Manages the complete lifecycle of performance monitoring, analysis,
    and intervention capabilities for the Kalshi trading system.
    """
    
    def __init__(self):
        self.logger = get_trading_logger("performance_system")
        self.scheduler = None
        self.running = False
        
    async def start_system(self, config: Optional[ScheduleConfig] = None) -> None:
        """Start the complete performance analysis system."""
        self.logger.info("🚀 Starting Kalshi Performance Analysis System")
        
        try:
            # Start the automated scheduler
            self.scheduler = start_performance_scheduler(config)
            self.running = True
            
            # Run initial health check
            initial_summary = await dashboard_get_summary()
            
            self.logger.info(
                "✅ Performance system started successfully",
                system_status=initial_summary['system_status'],
                health_score=initial_summary['metrics']['health_score'],
                critical_issues=initial_summary['metrics']['critical_issues']
            )
            
            # Display startup summary
            print(f"\n{'='*60}")
            print("🎯 KALSHI PERFORMANCE ANALYSIS SYSTEM ACTIVE")
            print(f"{'='*60}")
            print(f"System Status: {initial_summary['system_status']}")
            print(f"Health Score: {initial_summary['metrics']['health_score']:.1f}/100")
            print(f"Critical Issues: {initial_summary['metrics']['critical_issues']}")
            print(f"Warnings: {initial_summary['metrics']['warnings']}")
            print(f"Available Cash: ${initial_summary['metrics']['available_cash']:.2f}")
            print(f"Capital Utilization: {initial_summary['metrics']['capital_utilization']:.1f}%")
            print(f"Active Positions: {initial_summary['metrics']['active_positions']}")
            print(f"{'='*60}")
            
            # Show critical alerts if any
            if initial_summary['alerts']:
                print("\n🚨 CRITICAL ALERTS:")
                for alert in initial_summary['alerts']:
                    priority_icon = "🚨" if alert['priority'] == 'CRITICAL' else "⚠️"
                    print(f"{priority_icon} {alert['title']}")
                    print(f"   {alert['message']}")
                print(f"{'='*60}")
            
            # Show quick actions
            if initial_summary['quick_actions']:
                print("\n🎯 QUICK ACTIONS AVAILABLE:")
                for action in initial_summary['quick_actions']:
                    print(f"• {action['title']}")
                print(f"{'='*60}")
            
            print(f"\nScheduler running. Daily analysis: {config.daily_analysis_time if config else '09:00'}")
            print("Press Ctrl+C to stop the system\n")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start performance system: {e}")
            raise
    
    def stop_system(self) -> None:
        """Stop the performance analysis system."""
        self.logger.info("🛑 Stopping Performance Analysis System")
        
        if self.scheduler:
            stop_performance_scheduler()
            self.scheduler = None
        
        self.running = False
        self.logger.info("✅ Performance system stopped")
    
    async def run_immediate_analysis(self) -> Dict[str, Any]:
        """Run immediate comprehensive analysis."""
        self.logger.info("🔍 Running immediate performance analysis")
        
        try:
            report = await run_performance_analysis()
            
            print(f"\n{'='*60}")
            print("📊 IMMEDIATE ANALYSIS RESULTS")
            print(f"{'='*60}")
            print(f"Health Score: {report['summary']['overall_health_score']:.1f}/100")
            print(f"Critical Issues: {report['summary']['total_critical_issues']}")
            print(f"Warnings: {report['summary']['total_warnings']}")
            print(f"Action Items Generated: {len(report['action_items'])}")
            
            # Show critical action items
            critical_actions = [a for a in report['action_items'] if a['priority'] == 'CRITICAL']
            if critical_actions:
                print(f"\n🚨 CRITICAL ACTIONS REQUIRED:")
                for action in critical_actions:
                    print(f"• {action['action']}")
                    print(f"  Target: {action['target_date']}")
                    print(f"  Steps: {', '.join(action['implementation_steps'][:2])}...")
            
            high_actions = [a for a in report['action_items'] if a['priority'] == 'HIGH']
            if high_actions:
                print(f"\n🔶 HIGH PRIORITY ACTIONS:")
                for action in high_actions:
                    print(f"• {action['action']}")
            
            # Show Grok4 analysis summary
            if 'grok_analysis' in report and report['grok_analysis']['status'] == 'success':
                print(f"\n🤖 GROK4 ANALYSIS AVAILABLE:")
                analysis_preview = report['grok_analysis']['analysis_text'][:200]
                print(f"{analysis_preview}...")
                print(f"💰 Analysis Cost: ${report['grok_analysis']['cost']:.4f}")
            
            print(f"{'='*60}\n")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Analysis failed: {e}")
            raise
    
    async def show_system_status(self) -> None:
        """Display current system status."""
        try:
            summary = await dashboard_get_summary()
            
            print(f"\n{'='*60}")
            print("📊 SYSTEM STATUS")
            print(f"{'='*60}")
            print(f"Overall Status: {summary['system_status']} ({summary['status_color']})")
            print(f"Health Score: {summary['metrics']['health_score']:.1f}/100")
            print(f"Risk Level: {summary['performance_summary']['risk_level']}")
            print(f"Trending: {summary['metrics']['trending']}")
            
            print(f"\n📈 PORTFOLIO METRICS:")
            print(f"Available Cash: ${summary['metrics']['available_cash']:.2f}")
            print(f"Capital Utilization: {summary['metrics']['capital_utilization']:.1f}%")
            print(f"Active Positions: {summary['metrics']['active_positions']}")
            print(f"Win Rate: {summary['metrics']['win_rate']:.1%}")
            print(f"Total P&L: ${summary['metrics']['total_pnl']:.2f}")
            
            print(f"\n🚨 ISSUES:")
            print(f"Critical Issues: {summary['metrics']['critical_issues']}")
            print(f"Warnings: {summary['metrics']['warnings']}")
            
            print(f"\n⏰ SCHEDULER:")
            print(f"Running: {summary['scheduler']['running']}")
            print(f"Last Analysis: {summary['scheduler'].get('last_analysis', 'Never')}")
            print(f"Next Analysis: {summary['scheduler'].get('next_analysis', 'Unknown')}")
            
            if summary['alerts']:
                print(f"\n🚨 ACTIVE ALERTS:")
                for alert in summary['alerts'][:5]:  # Show top 5
                    priority_icon = "🚨" if alert['priority'] == 'CRITICAL' else "⚠️"
                    print(f"{priority_icon} {alert['title']}")
            
            print(f"{'='*60}\n")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get system status: {e}")
    
    async def emergency_intervention(self) -> None:
        """Run emergency analysis and intervention."""
        print(f"\n{'='*60}")
        print("🚨 EMERGENCY INTERVENTION MODE")
        print(f"{'='*60}")
        print("Running emergency analysis with Grok4...")
        
        try:
            emergency_result = await dashboard_trigger_analysis()
            
            print(f"\n📊 EMERGENCY ANALYSIS RESULTS:")
            print(f"Health Score: {emergency_result['health_score']:.1f}/100")
            print(f"Critical Issues: {emergency_result['critical_issues']}")
            
            if emergency_result.get('urgent_actions'):
                print(f"\n🚨 URGENT ACTIONS REQUIRED:")
                for action in emergency_result['urgent_actions']:
                    print(f"• {action['action']}")
                    print(f"  Rationale: {action['rationale']}")
                    print(f"  Target: {action['target_date']}")
                    if action['steps']:
                        print(f"  Immediate Steps:")
                        for step in action['steps']:
                            print(f"    - {step}")
            
            if emergency_result.get('immediate_steps'):
                print(f"\n⚡ IMMEDIATE STEPS:")
                for step in emergency_result['immediate_steps']:
                    print(f"• {step}")
            
            print(f"{'='*60}\n")
            
        except Exception as e:
            self.logger.error(f"❌ Emergency intervention failed: {e}")
    
    async def interactive_mode(self) -> None:
        """Run interactive command mode."""
        print(f"\n{'='*60}")
        print("🎯 INTERACTIVE PERFORMANCE MANAGEMENT")
        print(f"{'='*60}")
        print("Available commands:")
        print("  status - Show system status")
        print("  analyze - Run immediate analysis")
        print("  emergency - Emergency intervention")
        print("  alerts - Show critical alerts")
        print("  metrics - Show current metrics")
        print("  help - Show this help")
        print("  exit - Exit interactive mode")
        print(f"{'='*60}\n")
        
        while True:
            try:
                command = input("performance> ").strip().lower()
                
                if command == "exit":
                    break
                elif command == "status":
                    await self.show_system_status()
                elif command == "analyze":
                    await self.run_immediate_analysis()
                elif command == "emergency":
                    await self.emergency_intervention()
                elif command == "alerts":
                    alerts = await dashboard_get_alerts()
                    if alerts:
                        print("\n🚨 CRITICAL ALERTS:")
                        for alert in alerts:
                            priority_icon = "🚨" if alert['priority'] == 'CRITICAL' else "⚠️"
                            print(f"{priority_icon} {alert['title']}")
                            print(f"   {alert['message']}")
                            print(f"   Action: {alert['action_required']}")
                    else:
                        print("✅ No critical alerts")
                elif command == "metrics":
                    metrics = await dashboard_get_metrics()
                    print(f"\n📊 CURRENT METRICS:")
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            print(f"{key}: {value:.2f}")
                        else:
                            print(f"{key}: {value}")
                elif command == "help":
                    print("\nAvailable commands:")
                    print("  status, analyze, emergency, alerts, metrics, help, exit")
                elif command == "":
                    continue
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                
                print()  # Add spacing
                
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"❌ Command failed: {e}")
    
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
            self.stop_system()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point for the performance system manager."""
    # Setup logging
    setup_logging("INFO")
    
    parser = argparse.ArgumentParser(
        description="Kalshi Trading Performance Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start the automated system
  python performance_system_manager.py --start
  
  # Run immediate analysis
  python performance_system_manager.py --analyze
  
  # Show system status
  python performance_system_manager.py --status
  
  # Emergency intervention mode
  python performance_system_manager.py --emergency
  
  # Interactive mode
  python performance_system_manager.py --interactive
        """
    )
    
    parser.add_argument("--start", action="store_true", 
                       help="Start the automated performance analysis system")
    parser.add_argument("--analyze", action="store_true",
                       help="Run immediate comprehensive analysis")
    parser.add_argument("--status", action="store_true",
                       help="Show current system status")
    parser.add_argument("--emergency", action="store_true",
                       help="Run emergency intervention analysis")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive command mode")
    parser.add_argument("--daily-time", default="09:00",
                       help="Daily analysis time (HH:MM)")
    parser.add_argument("--weekly-day", default="monday",
                       help="Weekly analysis day")
    parser.add_argument("--health-threshold", type=float, default=50.0,
                       help="Health score threshold for alerts")
    
    args = parser.parse_args()
    
    # Create system manager
    manager = PerformanceSystemManager()
    
    try:
        if args.start:
            # Configure scheduler
            config = ScheduleConfig(
                daily_analysis_time=args.daily_time,
                weekly_deep_analysis_day=args.weekly_day,
                health_score_threshold=args.health_threshold
            )
            
            # Setup signal handlers for graceful shutdown
            manager.setup_signal_handlers()
            
            # Start the system
            await manager.start_system(config)
            
            # Keep running until interrupted
            try:
                while manager.running:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutting down...")
                manager.stop_system()
        
        elif args.analyze:
            await manager.run_immediate_analysis()
        
        elif args.status:
            await manager.show_system_status()
        
        elif args.emergency:
            await manager.emergency_intervention()
        
        elif args.interactive:
            await manager.interactive_mode()
        
        else:
            parser.print_help()
            print(f"\n💡 Quick start: python {parser.prog} --start")
    
    except Exception as e:
        print(f"❌ System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 