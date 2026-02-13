"""
Configuration settings for the Kalshi trading system.
Manages trading parameters, API configurations, and risk management settings.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class APIConfig:
    """API configuration settings."""
    kalshi_api_key: str = field(default_factory=lambda: os.getenv("KALSHI_API_KEY", ""))
    kalshi_base_url: str = "https://api.elections.kalshi.com"  # Updated to new API endpoint
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    xai_api_key: str = field(default_factory=lambda: os.getenv("XAI_API_KEY", ""))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openai_base_url: str = "https://api.openai.com/v1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"


@dataclass
class EnsembleConfig:
    """Multi-model ensemble configuration."""
    enabled: bool = True
    # Model roster for ensemble decisions
    models: Dict[str, Dict] = field(default_factory=lambda: {
        "grok-4-1-fast-reasoning": {"provider": "xai", "role": "forecaster", "weight": 0.30},
        "anthropic/claude-sonnet-4.5": {"provider": "openrouter", "role": "news_analyst", "weight": 0.20},
        "openai/o3": {"provider": "openrouter", "role": "bull_researcher", "weight": 0.20},
        "google/gemini-3-pro-preview": {"provider": "openrouter", "role": "bear_researcher", "weight": 0.15},
        "deepseek/deepseek-v3.2": {"provider": "openrouter", "role": "risk_manager", "weight": 0.15},
    })
    min_models_for_consensus: int = 3
    disagreement_threshold: float = 0.25  # Std dev above this = low confidence
    parallel_requests: bool = True
    debate_enabled: bool = True
    calibration_tracking: bool = True
    max_ensemble_cost: float = 0.50  # Max cost per ensemble decision


@dataclass
class SentimentConfig:
    """News and sentiment analysis configuration."""
    enabled: bool = True
    rss_feeds: List[str] = field(default_factory=lambda: [
        "https://feeds.reuters.com/reuters/topNews",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
        "https://feeds.bbci.co.uk/news/business/rss.xml",
    ])
    sentiment_model: str = "google/gemini-3-flash-preview"  # Fast/cheap for sentiment
    cache_ttl_minutes: int = 30
    max_articles_per_source: int = 10
    relevance_threshold: float = 0.3


# Trading strategy configuration - INCREASED AGGRESSIVENESS
@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    # Position sizing and risk management - MADE MORE AGGRESSIVE  
    max_position_size_pct: float = 5.0  # INCREASED: Back to 5% per position (was 3%)
    max_daily_loss_pct: float = 15.0    # INCREASED: Allow 15% daily loss (was 10%) 
    max_positions: int = 15              # INCREASED: Allow 15 concurrent positions (was 10)
    min_balance: float = 50.0           # REDUCED: Lower minimum to trade more (was 100)
    
    # Market filtering criteria - MUCH MORE PERMISSIVE
    min_volume: float = 200.0            # DECREASED: Much lower volume requirement (was 500, now 200)
    max_time_to_expiry_days: int = 30    # INCREASED: Allow longer timeframes (was 14, now 30)
    
    # AI decision making - MORE AGGRESSIVE THRESHOLDS
    min_confidence_to_trade: float = 0.50   # DECREASED: Lower confidence barrier (was 0.65, now 0.50)
    scan_interval_seconds: int = 30      # DECREASED: Scan more frequently (was 60, now 30)
    
    # AI model configuration
    primary_model: str = "grok-4-1-fast-reasoning"  # Latest xAI frontier reasoning model
    fallback_model: str = "grok-4-1-fast-non-reasoning"  # Fallback to non-reasoning variant
    ai_temperature: float = 0  # Lower temperature for more consistent JSON output
    ai_max_tokens: int = 8000    # Reasonable limit for reasoning models (grok-4 works better with 8000)
    
    # Position sizing (LEGACY - now using Kelly-primary approach)
    default_position_size: float = 3.0  # REDUCED: Now using Kelly Criterion as primary method (was 5%, now 3%)
    position_size_multiplier: float = 1.0  # Multiplier for AI confidence
    
    # Kelly Criterion settings (PRIMARY position sizing method) - MORE AGGRESSIVE
    use_kelly_criterion: bool = True        # Use Kelly Criterion for position sizing (PRIMARY METHOD)
    kelly_fraction: float = 0.75            # INCREASED: More aggressive Kelly multiplier (was 0.5, now 0.75)
    max_single_position: float = 0.05       # INCREASED: Higher position cap (was 0.03, now 5%)
    
    # Trading frequency - MORE FREQUENT
    market_scan_interval: int = 30          # DECREASED: Scan every 30 seconds (was 60)
    position_check_interval: int = 15       # DECREASED: Check positions every 15 seconds (was 30)
    max_trades_per_hour: int = 20           # INCREASED: Allow more trades per hour (was 10, now 20)
    run_interval_minutes: int = 10          # DECREASED: Run more frequently (was 15, now 10)
    num_processor_workers: int = 5      # Number of concurrent market processor workers
    
    # Market selection preferences
    preferred_categories: List[str] = field(default_factory=lambda: [])
    excluded_categories: List[str] = field(default_factory=lambda: [])
    
    # High-confidence, near-expiry strategy
    enable_high_confidence_strategy: bool = True
    high_confidence_threshold: float = 0.95  # LLM confidence needed
    high_confidence_market_odds: float = 0.90 # Market price to look for
    high_confidence_expiry_hours: int = 24   # Max hours until expiry

    # AI trading criteria - MORE PERMISSIVE
    max_analysis_cost_per_decision: float = 0.15  # INCREASED: Allow higher cost per decision (was 0.10, now 0.15)
    min_confidence_threshold: float = 0.45  # DECREASED: Lower confidence threshold (was 0.55, now 0.45)

    # Cost control and market analysis frequency - MORE PERMISSIVE
    daily_ai_budget: float = 10.0  # INCREASED: Higher daily budget (was 5.0, now 10.0)
    max_ai_cost_per_decision: float = 0.08  # INCREASED: Higher per-decision cost (was 0.05, now 0.08)
    analysis_cooldown_hours: int = 3  # DECREASED: Shorter cooldown (was 6, now 3)
    max_analyses_per_market_per_day: int = 4  # INCREASED: More analyses per day (was 2, now 4)
    
    # Daily AI spending limits - SAFETY CONTROLS
    daily_ai_cost_limit: float = 50.0  # Maximum daily spending on AI API calls (USD)
    enable_daily_cost_limiting: bool = True  # Enable daily cost limits
    sleep_when_limit_reached: bool = True  # Sleep until next day when limit reached

    # Enhanced market filtering to reduce analyses - MORE PERMISSIVE
    min_volume_for_ai_analysis: float = 200.0  # DECREASED: Much lower threshold (was 500, now 200)
    exclude_low_liquidity_categories: List[str] = field(default_factory=lambda: [
        # REMOVED weather and entertainment - trade all categories
    ])


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "DEBUG"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "logs/trading_system.log"
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    max_log_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


# BEAST MODE UNIFIED TRADING SYSTEM CONFIGURATION 🚀
# These settings control the advanced multi-strategy trading system

# === CAPITAL ALLOCATION ACROSS STRATEGIES ===
# Allocate capital across different trading approaches
market_making_allocation: float = 0.40  # 40% for market making (spread profits)
directional_allocation: float = 0.50    # 50% for directional trading (AI predictions) 
arbitrage_allocation: float = 0.10      # 10% for arbitrage opportunities

  # === PORTFOLIO OPTIMIZATION SETTINGS ===
# Kelly Criterion is now the PRIMARY position sizing method (moved to TradingConfig)
# total_capital: DYNAMICALLY FETCHED from Kalshi balance - never hardcoded!
use_risk_parity: bool = True            # Equal risk allocation vs equal capital
rebalance_hours: int = 6                # Rebalance portfolio every 6 hours
min_position_size: float = 5.0          # Minimum position size ($5 vs $10)
max_opportunities_per_batch: int = 50   # Limit opportunities to prevent optimization issues

# === RISK MANAGEMENT LIMITS ===
# Portfolio-level risk constraints (EXTREMELY RELAXED FOR TESTING)
max_volatility: float = 0.80            # Very high volatility allowed (80%)
max_correlation: float = 0.95           # Very high correlation allowed (95%)
max_drawdown: float = 0.50              # High drawdown tolerance (50%)
max_sector_exposure: float = 0.90       # Very high sector concentration (90%)

# === PERFORMANCE TARGETS ===
# System performance objectives - MORE AGGRESSIVE FOR MORE TRADES
target_sharpe: float = 0.3              # DECREASED: Lower Sharpe requirement (was 0.5, now 0.3)
target_return: float = 0.15             # INCREASED: Higher return target (was 0.10, now 0.15)
min_trade_edge: float = 0.08           # DECREASED: Lower edge requirement (was 0.15, now 8%)
min_confidence_for_large_size: float = 0.50  # DECREASED: Lower confidence requirement (was 0.65, now 50%)

# === DYNAMIC EXIT STRATEGIES ===
# Enhanced exit strategy settings - MORE AGGRESSIVE
use_dynamic_exits: bool = True
profit_threshold: float = 0.20          # DECREASED: Take profits sooner (was 0.25, now 0.20)
loss_threshold: float = 0.15            # INCREASED: Allow larger losses (was 0.10, now 0.15)
confidence_decay_threshold: float = 0.25  # INCREASED: Allow more confidence decay (was 0.20, now 0.25)
max_hold_time_hours: int = 240          # INCREASED: Hold longer (was 168, now 240 hours = 10 days)
volatility_adjustment: bool = True      # Adjust exits based on volatility

# === MARKET MAKING STRATEGY ===
# Settings for limit order market making - MORE AGGRESSIVE
enable_market_making: bool = True       # Enable market making strategy
min_spread_for_making: float = 0.01     # DECREASED: Accept smaller spreads (was 0.02, now 1¢)
max_inventory_risk: float = 0.15        # INCREASED: Allow higher inventory risk (was 0.10, now 15%)
order_refresh_minutes: int = 15         # Refresh orders every 15 minutes
max_orders_per_market: int = 4          # Maximum orders per market (2 each side)

# === MARKET SELECTION (ENHANCED FOR MORE OPPORTUNITIES) ===
# Removed time restrictions - trade ANY deadline with dynamic exits!
# max_time_to_expiry_days: REMOVED      # No longer used - trade any timeline!
min_volume_for_analysis: float = 200.0  # DECREASED: Much lower minimum volume (was 1000, now 200)
min_volume_for_market_making: float = 500.0  # DECREASED: Lower volume for market making (was 2000, now 500)
min_price_movement: float = 0.02        # DECREASED: Lower minimum range (was 0.05, now 2¢)
max_bid_ask_spread: float = 0.15        # INCREASED: Allow wider spreads (was 0.10, now 15¢)
min_confidence_long_term: float = 0.45  # DECREASED: Lower confidence for distant expiries (was 0.65, now 45%)

# === COST OPTIMIZATION (MORE GENEROUS) ===
# Enhanced cost controls for the beast mode system
daily_ai_budget: float = 15.0           # INCREASED: Higher budget for more opportunities (was 10.0, now 15.0)
max_ai_cost_per_decision: float = 0.12  # INCREASED: Higher per-decision limit (was 0.08, now 0.12)
analysis_cooldown_hours: int = 2        # DECREASED: Much shorter cooldown (was 4, now 2)
max_analyses_per_market_per_day: int = 6  # INCREASED: More analyses per day (was 3, now 6)
skip_news_for_low_volume: bool = True   # Skip expensive searches for low volume
news_search_volume_threshold: float = 1000.0  # News threshold

# === SYSTEM BEHAVIOR ===
# Overall system behavior settings
beast_mode_enabled: bool = True         # Enable the unified advanced system
fallback_to_legacy: bool = True         # Fallback to legacy system if needed
live_trading_enabled: bool = True       # Set to True for live trading
paper_trading_mode: bool = False        # Paper trading for testing
log_level: str = "INFO"                 # Logging level
performance_monitoring: bool = True     # Enable performance monitoring

# === ADVANCED FEATURES ===
# Cutting-edge features for maximum performance
cross_market_arbitrage: bool = False    # Enable when arbitrage module ready
multi_model_ensemble: bool = True       # Multi-model ensemble decisions (ENABLED)
sentiment_analysis: bool = True         # News sentiment analysis (ENABLED)
websocket_streaming: bool = True        # WebSocket real-time data (ENABLED)
options_strategies: bool = False        # Complex options strategies (future)
algorithmic_execution: bool = False     # Smart order execution (future)


@dataclass
class Settings:
    """Main settings class combining all configuration."""
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)

    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.api.kalshi_api_key:
            raise ValueError("KALSHI_API_KEY environment variable is required")

        if not self.api.xai_api_key:
            raise ValueError("XAI_API_KEY environment variable is required")

        if self.trading.max_position_size_pct <= 0 or self.trading.max_position_size_pct > 100:
            raise ValueError("max_position_size_pct must be between 0 and 100")

        if self.trading.min_confidence_to_trade <= 0 or self.trading.min_confidence_to_trade > 1:
            raise ValueError("min_confidence_to_trade must be between 0 and 1")

        return True


# Global settings instance
settings = Settings()

# Validate settings on import
try:
    settings.validate()
except ValueError as e:
    print(f"Configuration validation error: {e}")
    print("Please check your environment variables and configuration.") 