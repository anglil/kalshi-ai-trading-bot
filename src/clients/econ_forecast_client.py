"""
Multi-Source Economic Indicator Forecast Client

Fetches economic nowcast data from 3 free Federal Reserve sources:
  1. Atlanta Fed GDPNow — real-time GDP growth estimate
  2. Cleveland Fed Inflation Nowcast — CPI nowcast
  3. NY Fed Staff Nowcast — GDP and inflation nowcast

All sources are free and require no API keys.
"""

import asyncio
import aiohttp
import re
import ssl
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("econ_forecast_client")


@dataclass
class EconIndicator:
    """Configuration for a single economic indicator."""
    name: str                            # "CPI", "NFP", "GDP"
    kalshi_series_candidates: List[str]  # series tickers to try on Kalshi
    unit: str                            # "%", "K jobs", "%"
    cluster_tolerance: float             # consensus cluster tolerance
    sigma_high: float                    # sigma when high agreement
    sigma_med: float                     # sigma when medium agreement
    sigma_low: float                     # sigma when low agreement


# Indicator definitions
ECON_INDICATORS = {
    "CPI": EconIndicator(
        name="CPI",
        kalshi_series_candidates=["CPI", "KXCPI", "INFLATION", "CPIYOY"],
        unit="%",
        cluster_tolerance=0.1,
        sigma_high=0.05,
        sigma_med=0.1,
        sigma_low=0.2,
    ),
    "NFP": EconIndicator(
        name="NFP",
        kalshi_series_candidates=["NFP", "KXNFP", "JOBS", "NONFARM", "PAYROLLS"],
        unit="K",
        cluster_tolerance=25.0,
        sigma_high=15.0,
        sigma_med=30.0,
        sigma_low=60.0,
    ),
    "GDP": EconIndicator(
        name="GDP",
        kalshi_series_candidates=["GDP", "KXGDP", "GDPGROWTH"],
        unit="%",
        cluster_tolerance=0.3,
        sigma_high=0.15,
        sigma_med=0.3,
        sigma_low=0.6,
    ),
}


@dataclass
class ForecastSource:
    """A single economic forecast from one provider."""
    provider: str
    value: float          # forecast value in indicator units
    model_name: str
    indicator: str        # which indicator this is for


@dataclass
class MultiSourceForecast:
    """Aggregated economic forecasts from multiple providers."""
    indicator: str
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


async def _fetch_atlanta_fed_gdpnow(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    Fetch Atlanta Fed GDPNow real-time GDP estimate.
    Endpoint: Atlanta Fed GDPNow JSON data.
    """
    url = "https://www.atlantafed.org/cqer/research/gdpnow.aspx"
    alt_url = "https://www.atlantafed.org/-/media/documents/cqer/researchcq/gdpnow/RealGDPTrackingSlides.pdf"

    # Try the JSON feed first
    json_url = "https://www.atlantafed.org/cqer/research/gdpnow"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; EconBot/1.0)",
            "Accept": "text/html,application/json",
        }
        async with session.get(
            json_url,
            timeout=aiohttp.ClientTimeout(total=20),
            headers=headers,
            ssl=_make_ssl_ctx(),
            allow_redirects=True,
        ) as resp:
            if resp.status != 200:
                logger.warning(f"Atlanta Fed GDPNow returned {resp.status}")
                return None
            text = await resp.text()

        # Parse the GDPNow estimate from the page
        # Look for patterns like "X.X percent" or "X.X%"
        patterns = [
            r'GDPNow\s+model\s+estimate[^0-9]*?(-?\d+\.?\d*)\s*percent',
            r'latest\s+estimate[^0-9]*?(-?\d+\.?\d*)\s*percent',
            r'real\s+GDP\s+growth[^0-9]*?(-?\d+\.?\d*)\s*percent',
            r'(\-?\d+\.?\d*)\s*percent\s*(?:SAAR|annualized)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                return ForecastSource(
                    provider="atlanta_fed",
                    value=value,
                    model_name="Atlanta Fed GDPNow",
                    indicator="GDP",
                )

        logger.warning("Atlanta Fed GDPNow: could not parse GDP estimate from page")
        return None
    except Exception as e:
        logger.warning(f"Atlanta Fed GDPNow fetch failed: {e}")
        return None


async def _fetch_cleveland_fed_cpi(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    Fetch Cleveland Fed Inflation Nowcast.
    Provides a real-time CPI estimate.
    """
    url = "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; EconBot/1.0)",
            "Accept": "text/html,application/json",
        }
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=20),
            headers=headers,
            ssl=_make_ssl_ctx(),
            allow_redirects=True,
        ) as resp:
            if resp.status != 200:
                logger.warning(f"Cleveland Fed returned {resp.status}")
                return None
            text = await resp.text()

        # Parse CPI nowcast from page
        patterns = [
            r'CPI\s+(?:inflation\s+)?(?:of\s+)?(-?\d+\.?\d*)\s*percent',
            r'year-over-year[^0-9]*?(-?\d+\.?\d*)\s*percent',
            r'nowcast[^0-9]*?(-?\d+\.?\d*)\s*percent',
            r'(\d+\.?\d*)\s*%\s*(?:CPI|inflation)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                return ForecastSource(
                    provider="cleveland_fed",
                    value=value,
                    model_name="Cleveland Fed Inflation Nowcast",
                    indicator="CPI",
                )

        logger.warning("Cleveland Fed: could not parse CPI nowcast from page")
        return None
    except Exception as e:
        logger.warning(f"Cleveland Fed CPI fetch failed: {e}")
        return None


async def _fetch_ny_fed_nowcast(session: aiohttp.ClientSession, indicator: str = "GDP") -> Optional[ForecastSource]:
    """
    Fetch NY Fed Staff Nowcast for GDP growth.
    The NY Fed publishes a GDP nowcast weekly.
    """
    url = "https://www.newyorkfed.org/research/policy/nowcast"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; EconBot/1.0)",
            "Accept": "text/html,application/json",
        }
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=20),
            headers=headers,
            ssl=_make_ssl_ctx(),
            allow_redirects=True,
        ) as resp:
            if resp.status != 200:
                logger.warning(f"NY Fed Nowcast returned {resp.status}")
                return None
            text = await resp.text()

        # Parse GDP nowcast
        patterns = [
            r'GDP\s+growth[^0-9]*?(-?\d+\.?\d*)\s*(?:percent|%)',
            r'nowcast[^0-9]*?(-?\d+\.?\d*)\s*(?:percent|%)',
            r'Q\d\s+\d{4}[^0-9]*?(-?\d+\.?\d*)\s*(?:percent|%)',
            r'(\-?\d+\.?\d*)\s*(?:percent|%)\s*(?:SAAR|annualized)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                return ForecastSource(
                    provider="ny_fed",
                    value=value,
                    model_name="NY Fed Staff Nowcast",
                    indicator=indicator,
                )

        logger.warning("NY Fed: could not parse nowcast from page")
        return None
    except Exception as e:
        logger.warning(f"NY Fed nowcast fetch failed: {e}")
        return None


async def fetch_econ_forecasts(indicator: str) -> MultiSourceForecast:
    """
    Fetch economic forecasts for a given indicator from all available providers.

    For GDP: Atlanta Fed GDPNow + NY Fed Nowcast
    For CPI: Cleveland Fed Inflation Nowcast
    All sources are combined into consensus.
    """
    cache_key = f"econ_{indicator}"
    if cache_key in _cache:
        ts, cached = _cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return cached

    result = MultiSourceForecast(indicator=indicator)

    async with aiohttp.ClientSession() as session:
        tasks = []
        provider_names = []

        if indicator == "GDP":
            tasks.append(_fetch_atlanta_fed_gdpnow(session))
            provider_names.append("atlanta_fed")
            tasks.append(_fetch_ny_fed_nowcast(session, indicator="GDP"))
            provider_names.append("ny_fed")
        elif indicator == "CPI":
            tasks.append(_fetch_cleveland_fed_cpi(session))
            provider_names.append("cleveland_fed")
            tasks.append(_fetch_ny_fed_nowcast(session, indicator="CPI"))
            provider_names.append("ny_fed")
        elif indicator == "NFP":
            # NFP is hard to nowcast — use available macro sources
            tasks.append(_fetch_ny_fed_nowcast(session, indicator="NFP"))
            provider_names.append("ny_fed")
            tasks.append(_fetch_atlanta_fed_gdpnow(session))
            provider_names.append("atlanta_fed")
        else:
            logger.warning(f"Unknown indicator: {indicator}")
            return result

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
    values = [f"{s.value:.2f}{ECON_INDICATORS.get(indicator, EconIndicator('','',[],'',0,0,0)).unit}" for s in result.sources]
    logger.info(
        f"Econ forecasts ({indicator}): {len(result.sources)}/{len(provider_names)} sources — "
        f"{dict(zip(providers_ok, values))}, "
        f"failed: {result.failed_sources or 'none'}"
    )
    return result
