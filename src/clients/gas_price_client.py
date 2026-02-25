"""
Multi-Source Gas Price Client

Fetches national average gas price data from 3 free providers concurrently:
  1. EIA (Energy Information Administration) — weekly national average (needs free API key)
  2. AAA (American Automobile Association) — HTML scrape of national average
  3. GasBuddy — HTML scrape of national average

EIA requires a free API key (register at eia.gov). AAA and GasBuddy need no keys.
"""

import asyncio
import aiohttp
import os
import re
import ssl
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("gas_price_client")


@dataclass
class ForecastSource:
    """A single gas price estimate from one provider."""
    provider: str
    value: float          # price in $/gallon
    model_name: str


@dataclass
class MultiSourceForecast:
    """Aggregated gas price data from multiple providers."""
    target: str           # e.g. "national_gas_price"
    sources: List[ForecastSource] = field(default_factory=list)
    failed_sources: List[str] = field(default_factory=list)


# In-memory cache: key -> (timestamp, MultiSourceForecast)
_cache: Dict[str, Tuple[float, MultiSourceForecast]] = {}
_CACHE_TTL = 1800  # 30 minutes


def _make_ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def _fetch_eia(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    Fetch weekly national average gas price from EIA API.
    Requires EIA_API_KEY environment variable.
    """
    api_key = os.getenv("EIA_API_KEY", "")
    if not api_key:
        logger.warning("EIA_API_KEY not set — skipping EIA source")
        return None

    url = (
        f"https://api.eia.gov/v2/petroleum/pri/gnd/data/"
        f"?api_key={api_key}"
        f"&frequency=weekly"
        f"&data[0]=value"
        f"&facets[product][]=EPMR"
        f"&facets[duoarea][]=NUS"
        f"&sort[0][column]=period"
        f"&sort[0][direction]=desc"
        f"&length=1"
    )
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=15),
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"EIA API returned {resp.status}")
                return None
            data = await resp.json()

        records = data.get("response", {}).get("data", [])
        if not records:
            logger.warning("EIA API: no data records returned")
            return None

        price = float(records[0].get("value", 0))
        if price <= 0:
            return None

        return ForecastSource(
            provider="eia",
            value=price,
            model_name="EIA Weekly National Average",
        )
    except Exception as e:
        logger.warning(f"EIA fetch failed: {e}")
        return None


async def _fetch_aaa_gas(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    Fetch current national average gas price from AAA by scraping HTML.
    The first dollar amount on the page is the national regular average.
    """
    url = "https://gasprices.aaa.com/"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html",
        }
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=15),
            headers=headers,
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"AAA gas prices returned {resp.status}")
                return None
            html = await resp.text()

        # Extract dollar amounts from page — first match is national regular average
        prices = re.findall(r'\$(\d+\.\d{2,3})', html)
        if not prices:
            logger.warning("AAA: no gas prices found in HTML")
            return None

        price = float(prices[0])
        if price <= 0 or price > 10:
            return None

        return ForecastSource(
            provider="aaa",
            value=price,
            model_name="AAA National Average",
        )
    except Exception as e:
        logger.warning(f"AAA gas fetch failed: {e}")
        return None


async def _fetch_gasbuddy(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    Fetch current national average gas price from GasBuddy by scraping HTML.
    """
    url = "https://www.gasbuddy.com/charts"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html",
        }
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=15),
            headers=headers,
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"GasBuddy returned {resp.status}")
                return None
            html = await resp.text()

        # Look for national average price pattern
        prices = re.findall(r'\$(\d+\.\d{2,3})', html)
        if not prices:
            logger.warning("GasBuddy: no gas prices found in HTML")
            return None

        price = float(prices[0])
        if price <= 0 or price > 10:
            return None

        return ForecastSource(
            provider="gasbuddy",
            value=price,
            model_name="GasBuddy National Average",
        )
    except Exception as e:
        logger.warning(f"GasBuddy gas fetch failed: {e}")
        return None


async def fetch_gas_forecasts() -> MultiSourceForecast:
    """
    Fetch gas price data from all 3 providers concurrently.
    Returns a MultiSourceForecast with whatever sources succeeded.
    """
    cache_key = "national_gas"
    if cache_key in _cache:
        ts, cached = _cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return cached

    result = MultiSourceForecast(target="national_gas_price")

    async with aiohttp.ClientSession() as session:
        eia_task = _fetch_eia(session)
        aaa_task = _fetch_aaa_gas(session)
        gasbuddy_task = _fetch_gasbuddy(session)

        outcomes = await asyncio.gather(eia_task, aaa_task, gasbuddy_task, return_exceptions=True)

    provider_names = ["eia", "aaa", "gasbuddy"]
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
    prices = [f"${s.value:.3f}" for s in result.sources]
    logger.info(
        f"Gas price data: {len(result.sources)}/3 sources — "
        f"{dict(zip(providers_ok, prices))}, "
        f"failed: {result.failed_sources or 'none'}"
    )
    return result
