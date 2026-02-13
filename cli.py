#!/usr/bin/env python3
"""
Kalshi AI Trading Bot -- Unified CLI

Provides a single entry point for all bot operations:
    python cli.py run          Start the trading bot
    python cli.py dashboard    Launch the Streamlit monitoring dashboard
    python cli.py status       Show portfolio balance, positions, and P&L
    python cli.py backtest     Run backtests (placeholder)
    python cli.py health       Verify API connections, database, and configuration
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """Start the Beast Mode trading bot."""
    from src.utils.logging_setup import setup_logging
    from beast_mode_bot import BeastModeBot

    log_level = getattr(args, "log_level", "INFO")
    setup_logging(log_level=log_level)

    live = getattr(args, "live", False)
    paper = getattr(args, "paper", False)

    # --paper explicitly forces paper mode; --live forces live mode.
    # If neither is given, default is paper trading.
    if live and paper:
        print("Error: --live and --paper are mutually exclusive.")
        sys.exit(1)

    live_mode = live and not paper

    if live_mode:
        print("WARNING: LIVE TRADING MODE ENABLED")
        print("This will use real money and place actual trades.")

    bot = BeastModeBot(live_mode=live_mode)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user.")


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch the Streamlit monitoring dashboard."""
    import subprocess

    # Prefer the dedicated dashboard launch script if it exists.
    dashboard_script = Path(__file__).parent / "scripts" / "launch_dashboard.py"
    beast_dashboard = Path(__file__).parent / "beast_mode_dashboard.py"

    if dashboard_script.exists():
        subprocess.run([sys.executable, str(dashboard_script)], check=False)
    elif beast_dashboard.exists():
        # Fall back to running the dashboard module directly.
        from src.utils.logging_setup import setup_logging
        from beast_mode_bot import BeastModeBot

        setup_logging(log_level="INFO")
        bot = BeastModeBot(live_mode=False, dashboard_mode=True)
        try:
            asyncio.run(bot.run())
        except KeyboardInterrupt:
            print("\nDashboard stopped by user.")
    else:
        print("Error: No dashboard script found.")
        sys.exit(1)


def cmd_status(args: argparse.Namespace) -> None:
    """Show current portfolio status: balance, positions, and P&L."""

    async def _status() -> None:
        from src.clients.kalshi_client import KalshiClient

        client = KalshiClient()
        try:
            # Fetch balance
            balance_resp = await client.get_balance()
            balance_cents = balance_resp.get("balance", 0)
            balance_usd = balance_cents / 100.0

            # Fetch positions
            positions_resp = await client.get_positions()
            positions = positions_resp.get("market_positions", [])

            # Display
            print("=" * 56)
            print("  PORTFOLIO STATUS")
            print("=" * 56)
            print(f"  Available Balance:  ${balance_usd:>10,.2f}")
            print(f"  Open Positions:     {len(positions):>10}")

            total_cost = 0.0
            total_market_value = 0.0

            if positions:
                print()
                print(f"  {'Ticker':<20} {'Side':<6} {'Qty':>5} {'Avg':>7} {'Value':>9}")
                print(f"  {'-'*20} {'-'*6} {'-'*5} {'-'*7} {'-'*9}")

                for pos in positions:
                    ticker = pos.get("ticker", "???")
                    # Kalshi positions may use different field names
                    side = "YES" if pos.get("position", 0) > 0 else "NO"
                    qty = abs(pos.get("position", pos.get("total_traded", 0)))
                    avg_price = pos.get("average_price", 0)
                    if isinstance(avg_price, (int, float)) and avg_price > 1:
                        avg_price = avg_price / 100.0  # convert cents to dollars
                    market_value = qty * avg_price
                    total_cost += market_value
                    total_market_value += market_value
                    print(
                        f"  {ticker:<20} {side:<6} {qty:>5} "
                        f"${avg_price:>5.2f} ${market_value:>7.2f}"
                    )

            print()
            print(f"  Position Cost:      ${total_cost:>10,.2f}")
            print(f"  Total Portfolio:    ${balance_usd + total_cost:>10,.2f}")
            print("=" * 56)
        finally:
            await client.close()

    try:
        asyncio.run(_status())
    except Exception as exc:
        print(f"Error fetching status: {exc}")
        sys.exit(1)


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run backtests (placeholder)."""
    print("=" * 56)
    print("  BACKTESTING")
    print("=" * 56)
    print()
    print("  Backtesting engine coming soon.")
    print()
    print("  Planned features:")
    print("    - Historical market replay")
    print("    - Strategy parameter optimization")
    print("    - Walk-forward analysis")
    print("    - Monte Carlo simulation")
    print()
    print("=" * 56)


def cmd_health(args: argparse.Namespace) -> None:
    """Run health checks on configuration, API, and database."""

    checks_passed = 0
    checks_failed = 0

    def ok(label: str, detail: str = "") -> None:
        nonlocal checks_passed
        checks_passed += 1
        suffix = f" -- {detail}" if detail else ""
        print(f"  [PASS] {label}{suffix}")

    def fail(label: str, detail: str = "") -> None:
        nonlocal checks_failed
        checks_failed += 1
        suffix = f" -- {detail}" if detail else ""
        print(f"  [FAIL] {label}{suffix}")

    print("=" * 56)
    print("  HEALTH CHECK")
    print("=" * 56)
    print()

    # 1. .env file
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        ok(".env file exists")
    else:
        fail(".env file missing", "copy env.template to .env and fill in keys")

    # 2. Required environment variables
    from dotenv import load_dotenv
    load_dotenv()

    for var in ("KALSHI_API_KEY", "XAI_API_KEY"):
        val = os.getenv(var, "")
        if val and val not in ("", "your_kalshi_api_key_here", "your_xai_api_key_here"):
            ok(f"{var} is set")
        else:
            fail(f"{var} is missing or placeholder")

    # 3. Kalshi API connection
    async def _check_api() -> None:
        from src.clients.kalshi_client import KalshiClient
        client = KalshiClient()
        try:
            balance_resp = await client.get_balance()
            balance_usd = balance_resp.get("balance", 0) / 100.0
            ok("Kalshi API connection", f"balance=${balance_usd:,.2f}")
        except Exception as exc:
            fail("Kalshi API connection", str(exc))
        finally:
            await client.close()

    try:
        asyncio.run(_check_api())
    except Exception as exc:
        fail("Kalshi API connection", str(exc))

    # 4. Database
    db_path = Path(__file__).parent / "trading_system.db"
    try:
        import aiosqlite

        async def _check_db() -> None:
            from src.utils.database import DatabaseManager
            db_manager = DatabaseManager()
            await db_manager.initialize()
            ok("Database initialization", str(db_path))

        asyncio.run(_check_db())
    except Exception as exc:
        fail("Database initialization", str(exc))

    # 5. Python version
    if sys.version_info >= (3, 12):
        ok("Python version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    else:
        fail("Python version", f"requires >=3.12, found {sys.version}")

    # Summary
    print()
    total = checks_passed + checks_failed
    print(f"  {checks_passed}/{total} checks passed")
    if checks_failed:
        print(f"  {checks_failed} issue(s) need attention")
    else:
        print("  All systems operational.")
    print("=" * 56)

    if checks_failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kalshi-bot",
        description="Kalshi AI Trading Bot -- Multi-model AI trading for prediction markets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python cli.py run --paper        Start in paper-trading mode\n"
            "  python cli.py run --live          Start in live-trading mode\n"
            "  python cli.py dashboard           Open the monitoring dashboard\n"
            "  python cli.py status              Check portfolio balance and positions\n"
            "  python cli.py health              Verify all connections and config\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = subparsers.add_parser(
        "run",
        help="Start the trading bot",
        description="Launch the Beast Mode trading bot with market making, directional trading, and portfolio optimization.",
    )
    mode_group = p_run.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading with real capital (default: paper trading)",
    )
    mode_group.add_argument(
        "--paper",
        action="store_true",
        help="Run in paper-trading mode (no real orders)",
    )
    p_run.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging verbosity (default: INFO)",
    )
    p_run.set_defaults(func=cmd_run)

    # --- dashboard ---
    p_dash = subparsers.add_parser(
        "dashboard",
        help="Launch the Streamlit monitoring dashboard",
        description="Open a real-time web dashboard showing portfolio performance, positions, risk metrics, and AI decision logs.",
    )
    p_dash.set_defaults(func=cmd_dashboard)

    # --- status ---
    p_status = subparsers.add_parser(
        "status",
        help="Show portfolio balance, positions, and P&L",
        description="Connect to the Kalshi API and display current account balance, open positions, and estimated portfolio value.",
    )
    p_status.set_defaults(func=cmd_status)

    # --- backtest ---
    p_bt = subparsers.add_parser(
        "backtest",
        help="Run backtests (coming soon)",
        description="Backtest trading strategies against historical market data. This feature is under development.",
    )
    p_bt.set_defaults(func=cmd_backtest)

    # --- health ---
    p_health = subparsers.add_parser(
        "health",
        help="Verify API connections, database, and configuration",
        description="Run a series of diagnostic checks: .env presence, API key configuration, Kalshi API connectivity, database initialization, and Python version.",
    )
    p_health.set_defaults(func=cmd_health)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
