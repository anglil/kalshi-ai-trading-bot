"""
Generate a rich interactive trading performance dashboard from live Kalshi data.
Mirrors what the Streamlit dashboard at localhost:8501 would show.
"""

import json
import asyncio
import sys
import os
from datetime import datetime, timezone
from collections import defaultdict
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo('US/Pacific')

sys.path.insert(0, '.')
from src.clients.kalshi_client import KalshiClient
