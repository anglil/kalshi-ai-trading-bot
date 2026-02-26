"""
Multi-Source Gas Price Client

Fetches national average gas price data from 3 independent free sources:
  1. FRED (St. Louis Fed) — EIA weekly national average via CSV (no key needed)
  2. AAA (American Automobile Association) — HTML scrape of national average
  3. EIA Public — HTML scrape of EIA Gas Diesel page (no key needed)

All 3 sources require no API keys. 3 sources enables majority-vote consensus.
"""

import asyncio
import aiohttp
import csv
import io
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


async def _fetch_fred_gas(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    Fetch weekly national average gas price from FRED CSV download.
    Series GASREGW = U.S. Regular All Formulations Gas Price, Weekly.
    This is the same EIA data served as a free CSV — no API key needed.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GASREGW"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; GasBot/1.0)",
            "Accept": "text/csv,*/*",
        }
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=15),
            headers=headers,
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"FRED gas price CSV returned {resp.status}")
                return None
            text = await resp.text()

        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            logger.warning("FRED gas: no data rows in CSV")
            return None

        last = rows[-1]
        price_str = last.get("GASREGW", "")
        if not price_str or price_str == ".":
            logger.warning("FRED gas: latest value is missing")
            return None

        price = float(price_str)
        if price <= 0 or price > 10:
            return None

        return ForecastSource(
            provider="fred",
            value=price,
            model_name="FRED/EIA Weekly National Average",
        )
    except Exception as e:
        logger.warning(f"FRED gas fetch failed: {e}")
        return None


async def _fetch_aaa_gas(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    Fetch current national average gas price from AAA by scraping HTML.
    The first dollar amount on the page is the national regular average.
    """
    url = "https://gasprices.aaa.com/"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://gasprices.aaa.com/",
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


async def _fetch_eia_public(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    Fetch national average gas price from EIA public web page (no API key needed).
    Scrapes the Gas Diesel page which shows current national regular average.
    """
    url = "https://www.eia.gov/petroleum/gasdiesel/"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.eia.gov/petroleum/",
        }
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=15),
            headers=headers,
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"EIA public page returned {resp.status}")
                return None
            html = await resp.text()

        # First dollar amount on the page is the national regular average
        prices = re.findall(r'\$(\d+\.\d{2,3})', html)
        if not prices:
            logger.warning("EIA public: no gas prices found in HTML")
            return None

        price = float(prices[0])
        if price <= 0 or price > 10:
            return None

        return ForecastSource(
            provider="eia_public",
            value=price,
            model_name="EIA Public Gas Price",
        )
    except Exception as e:
        logger.warning(f"EIA public fetch failed: {e}")
        return None


async def fetch_gas_forecasts() -> MultiSourceForecast:
    """
    Fetch gas price data from all 3 providers concurrently.
    Returns a MultiSourceForecast with whatever sources succeeded.
    3 sources enables majority-vote consensus (2/3 agree = medium, 3/3 = high).
    """
    cache_key = "national_gas"
    if cache_key in _cache:
        ts, cached = _cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return cached

    result = MultiSourceForecast(target="national_gas_price")

    async with aiohttp.ClientSession() as session:
        fred_task = _fetch_fred_gas(session)
        aaa_task = _fetch_aaa_gas(session)
        eia_pub_task = _fetch_eia_public(session)

        outcomes = await asyncio.gather(fred_task, aaa_task, eia_pub_task, return_exceptions=True)

    provider_names = ["fred", "aaa", "eia_public"]
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
