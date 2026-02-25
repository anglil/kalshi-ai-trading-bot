"""
Multi-Source Gas Price Client

Fetches national average gas price data from 3 free providers concurrently:
  1. EIA (Energy Information Administration) — weekly national average
  2. FRED (Federal Reserve Economic Data) — GASREGW series
  3. AAA (American Automobile Association) — current national average

EIA requires a free API key (register at eia.gov). FRED and AAA are free.
"""

import asyncio
import aiohttp
import os
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
    Series: PET.EMM_EPMR_PTE_NUS_DPG.W (regular gasoline, national)
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


async def _fetch_fred_gas(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    Fetch gas price from FRED GASREGW series.
    Uses the FRED API with the public DEMO key (rate-limited but free).
    """
    api_key = os.getenv("FRED_API_KEY", "DEMO")
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id=GASREGW"
        f"&api_key={api_key}"
        f"&file_type=json"
        f"&sort_order=desc"
        f"&limit=1"
    )
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=15),
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"FRED API returned {resp.status}")
                return None
            data = await resp.json()

        observations = data.get("observations", [])
        if not observations:
            logger.warning("FRED GASREGW: no observations returned")
            return None

        value_str = observations[0].get("value", ".")
        if value_str == ".":
            return None

        price = float(value_str)
        if price <= 0:
            return None

        return ForecastSource(
            provider="fred",
            value=price,
            model_name="FRED GASREGW",
        )
    except Exception as e:
        logger.warning(f"FRED gas fetch failed: {e}")
        return None


async def _fetch_aaa_gas(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    Fetch current national average gas price from AAA.
    Scrapes the AAA gas prices API endpoint.
    """
    url = "https://gasprices.aaa.com/wp-json/fuel/v1/prices"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; GasPriceBot/1.0)",
            "Accept": "application/json",
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
            data = await resp.json()

        # AAA returns structure with national averages
        # Try common response shapes
        regular = None
        if isinstance(data, dict):
            # Try direct access
            regular = data.get("regular")
            if regular is None:
                # Try nested under "national"
                national = data.get("national", {})
                if isinstance(national, dict):
                    regular = national.get("regular") or national.get("gas_regular")
                # Try under "today"
                if regular is None:
                    today = data.get("today", {})
                    if isinstance(today, dict):
                        regular = today.get("regular")

        if regular is None:
            logger.warning("AAA: could not extract regular gas price from response")
            return None

        price = float(regular)
        if price <= 0:
            return None

        return ForecastSource(
            provider="aaa",
            value=price,
            model_name="AAA National Average",
        )
    except Exception as e:
        logger.warning(f"AAA gas fetch failed: {e}")
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
        fred_task = _fetch_fred_gas(session)
        aaa_task = _fetch_aaa_gas(session)

        outcomes = await asyncio.gather(eia_task, fred_task, aaa_task, return_exceptions=True)

    provider_names = ["eia", "fred", "aaa"]
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
