"""
Multi-agent ensemble decision engine for Kalshi trading.

This package implements a multi-model AI ensemble that combines
forecasting, news analysis, bull/bear research, risk management,
and final trade execution into a structured decision pipeline.
"""

from src.agents.base_agent import BaseAgent
from src.agents.forecaster_agent import ForecasterAgent
from src.agents.news_analyst_agent import NewsAnalystAgent
from src.agents.bull_researcher import BullResearcher
from src.agents.bear_researcher import BearResearcher
from src.agents.risk_manager_agent import RiskManagerAgent
from src.agents.trader_agent import TraderAgent
from src.agents.ensemble import EnsembleRunner
from src.agents.debate import DebateRunner

__all__ = [
    "BaseAgent",
    "ForecasterAgent",
    "NewsAnalystAgent",
    "BullResearcher",
    "BearResearcher",
    "RiskManagerAgent",
    "TraderAgent",
    "EnsembleRunner",
    "DebateRunner",
]
