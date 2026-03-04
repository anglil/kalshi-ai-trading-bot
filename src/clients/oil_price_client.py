"""
Multi-Source WTI Crude Oil Price Forecast Client

Fetches 3 independent sources for consensus on WTI crude oil prices:

1. EIA STEO API — Short-Term Energy Outlook from the U.S. Energy Information
   Administration. Monthly WTI spot price forecasts. Free, no key needed.
   Endpoint: https://api.eia.gov/v2/steo/data/

2. Yahoo Finance CL=F — WTI Crude Oil Futures front-month contract price.
   Real-time market-implied forward price. Free via yfinance or direct API.

3. FRED DCOILWTICO — Daily WTI spot price from FRED (Federal Reserve).
   Combined with historical volatility to produce a probabilistic forecast.
   Free CSV download, no key needed.

All sources produce point estimates ($/barrel) that are converted to
probability distributions over Kalshi price brackets using a Gaussian model.
"""

import asyncio
import aiohttp
import csv
import io
import json
import math
import re
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("oil_price_client")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OilPriceForecast:
    """A single oil price forecast from one provider."""
    provider: str
    price: float          # Forecast price in $/barrel
    model_name: str
    horizon: str          # "spot", "1m", "3m" etc.
    volatility: Optional[float] = None  # Annualized vol if available


@dataclass
class MultiSourceOilForecast:
    """Aggregated oil price forecasts from multiple providers."""
    sources: List[OilPriceForecast] = field(default_factory=list)
    failed_sources: List[str] = field(default_factory=list)


# In-memory cache
_cache: Dict[str, Tuple[float, MultiSourceOilForecast]] = {}
_CACHE_TTL = 1800  # 30 minutes


def _make_ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ---------------------------------------------------------------------------
# Source 1: FRED MCOILWTICO — Monthly WTI Crude Oil Average
# ---------------------------------------------------------------------------

async def _fetch_fred_monthly_oil(
    session: aiohttp.ClientSession,
) -> Optional[OilPriceForecast]:
    """
    Fetch monthly average WTI crude oil price from FRED.

    MCOILWTICO is the monthly average Cushing, OK WTI Spot Price ($/barrel).
    This provides a smoothed view compared to the daily spot price.
    """
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MCOILWTICO"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; OilBot/1.0)",
            "Accept": "text/csv,*/*",
        }

        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=20),
            headers=headers,
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"FRED MCOILWTICO returned {resp.status}")
                return None
            text = await resp.text()

        if "<html" in text[:500].lower():
            logger.warning("FRED MCOILWTICO: got HTML instead of CSV")
            return None

        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            logger.warning("FRED MCOILWTICO: no data rows")
            return None

        valid_rows = [
            r for r in rows
            if r.get("MCOILWTICO", ".") not in (".", "", "-")
        ]

        if not valid_rows:
            return None

        latest_price = float(valid_rows[-1]["MCOILWTICO"])
        latest_date = valid_rows[-1].get("DATE", "?")

        # Compute monthly volatility
        volatility = None
        if len(valid_rows) >= 13:
            recent = valid_rows[-13:]
            prices = [float(r["MCOILWTICO"]) for r in recent]
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0:
                    returns.append(math.log(prices[i] / prices[i-1]))
            if returns:
                monthly_vol = (sum(r**2 for r in returns) / len(returns)) ** 0.5
                volatility = monthly_vol * math.sqrt(12)  # Annualize

        logger.info(
            f"FRED MCOILWTICO: ${latest_price:.2f}/bbl (date={latest_date})"
            + (f", 12m vol={volatility:.1%}" if volatility else "")
        )

        return OilPriceForecast(
            provider="fred_monthly_oil",
            price=round(latest_price, 2),
            model_name="FRED Monthly WTI Average (MCOILWTICO)",
            horizon="1m",
            volatility=volatility,
        )

    except Exception as e:
        logger.warning(f"FRED MCOILWTICO fetch failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Source 2: Yahoo Finance CL=F — WTI Futures
# ---------------------------------------------------------------------------

async def _fetch_yahoo_oil_futures(
    session: aiohttp.ClientSession,
) -> Optional[OilPriceForecast]:
    """
    Fetch WTI crude oil futures price from Yahoo Finance.

    CL=F is the front-month WTI crude oil futures contract.
    This represents the market's best estimate of near-term oil prices.
    """
    try:
        # Yahoo Finance v8 API for CL=F
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/CL=F"
            "?interval=1d&range=5d"
        )
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }

        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=20),
            headers=headers,
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"Yahoo Finance CL=F returned {resp.status}")
                return None
            data = await resp.json()

        chart = data.get("chart", {}).get("result", [])
        if not chart:
            logger.warning("Yahoo Finance CL=F: no chart data")
            return None

        meta = chart[0].get("meta", {})
        price = meta.get("regularMarketPrice", 0)

        if not price or price <= 0:
            # Try from indicators
            indicators = chart[0].get("indicators", {}).get("quote", [{}])
            if indicators:
                closes = indicators[0].get("close", [])
                valid_closes = [c for c in closes if c is not None]
                if valid_closes:
                    price = valid_closes[-1]

        if not price or price <= 0:
            logger.warning("Yahoo Finance CL=F: could not extract price")
            return None

        # Calculate recent volatility from daily returns
        volatility = None
        indicators = chart[0].get("indicators", {}).get("quote", [{}])
        if indicators:
            closes = indicators[0].get("close", [])
            valid_closes = [c for c in closes if c is not None]
            if len(valid_closes) >= 3:
                returns = []
                for i in range(1, len(valid_closes)):
                    if valid_closes[i-1] > 0:
                        returns.append(
                            math.log(valid_closes[i] / valid_closes[i-1])
                        )
                if returns:
                    daily_vol = (sum(r**2 for r in returns) / len(returns)) ** 0.5
                    volatility = daily_vol * math.sqrt(252)  # Annualize

        logger.info(
            f"Yahoo Finance CL=F: ${price:.2f}/bbl"
            + (f", vol={volatility:.1%}" if volatility else "")
        )

        return OilPriceForecast(
            provider="yahoo_clf",
            price=round(price, 2),
            model_name="Yahoo Finance WTI Futures (CL=F)",
            horizon="front_month",
            volatility=volatility,
        )

    except Exception as e:
        logger.warning(f"Yahoo Finance CL=F fetch failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Source 3: FRED DCOILWTICO — WTI Spot Price
# ---------------------------------------------------------------------------

async def _fetch_fred_oil(
    session: aiohttp.ClientSession,
) -> Optional[OilPriceForecast]:
    """
    Fetch WTI crude oil spot price from FRED.

    DCOILWTICO is the daily Cushing, OK WTI Spot Price ($/barrel).
    We also compute 20-day historical volatility for the Gaussian model.
    """
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; OilBot/1.0)",
            "Accept": "text/csv,*/*",
        }

        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=20),
            headers=headers,
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"FRED DCOILWTICO returned {resp.status}")
                return None
            text = await resp.text()

        if "<html" in text[:500].lower():
            logger.warning("FRED DCOILWTICO: got HTML instead of CSV")
            return None

        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            logger.warning("FRED DCOILWTICO: no data rows")
            return None

        # Get valid (non-missing) values from the last 30 rows
        valid_rows = [
            r for r in rows
            if r.get("DCOILWTICO", ".") not in (".", "", "-")
        ]

        if not valid_rows:
            return None

        latest_price = float(valid_rows[-1]["DCOILWTICO"])
        latest_date = valid_rows[-1].get("DATE", "?")

        # Compute 20-day historical volatility
        volatility = None
        if len(valid_rows) >= 21:
            recent = valid_rows[-21:]
            prices = [float(r["DCOILWTICO"]) for r in recent]
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0:
                    returns.append(math.log(prices[i] / prices[i-1]))
            if returns:
                daily_vol = (sum(r**2 for r in returns) / len(returns)) ** 0.5
                volatility = daily_vol * math.sqrt(252)

        logger.info(
            f"FRED DCOILWTICO: ${latest_price:.2f}/bbl (date={latest_date})"
            + (f", 20d vol={volatility:.1%}" if volatility else "")
        )

        return OilPriceForecast(
            provider="fred_oil",
            price=round(latest_price, 2),
            model_name="FRED WTI Spot (DCOILWTICO)",
            horizon="spot",
            volatility=volatility,
        )

    except Exception as e:
        logger.warning(f"FRED DCOILWTICO fetch failed: {e}")
        return None


# ===========================================================================
# Dispatch — 3 sources fetched concurrently
# ===========================================================================

async def fetch_oil_price_forecasts() -> MultiSourceOilForecast:
    """
    Fetch WTI crude oil price forecasts from 3 independent sources.

    1. FRED MCOILWTICO — monthly average WTI price
    2. Yahoo Finance CL=F — market-implied futures price
    3. FRED DCOILWTICO — daily spot price with historical volatility

    3 sources enables majority-vote consensus (2/3 agree = medium, 3/3 = high).
    """
    cache_key = "oil_price"
    if cache_key in _cache:
        ts, cached = _cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return cached

    result = MultiSourceOilForecast()

    async with aiohttp.ClientSession() as session:
        tasks = [
            _fetch_fred_monthly_oil(session),
            _fetch_yahoo_oil_futures(session),
            _fetch_fred_oil(session),
        ]
        provider_names = ["fred_monthly_oil", "yahoo_clf", "fred_oil"]

        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    for name, outcome in zip(provider_names, outcomes):
        if isinstance(outcome, Exception):
            logger.warning(f"Provider {name} raised exception: {outcome}")
            result.failed_sources.append(name)
        elif outcome is None:
            result.failed_sources.append(name)
        else:
            result.sources.append(outcome)

    _cache[cache_key] = (time.time(), result)

    providers_ok = [s.provider for s in result.sources]
    prices = [f"${s.price:.2f}" for s in result.sources]
    logger.info(
        f"Oil price forecasts: {len(result.sources)}/3 sources — "
        f"{dict(zip(providers_ok, prices))}, "
        f"failed: {result.failed_sources or 'none'}"
    )
    return result
