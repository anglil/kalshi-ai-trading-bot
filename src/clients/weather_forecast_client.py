"""
Multi-Source Weather Forecast Client

Fetches temperature forecasts from 3 free providers concurrently:
  1. NWS (National Weather Service) — via existing nws_client.py
  2. Open-Meteo GFS — NOAA's Global Forecast System
  3. Open-Meteo ECMWF — European Centre model (IFS 0.25°)

All sources are free and require no API keys.
"""

import asyncio
import aiohttp
import ssl
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.clients.nws_client import (
    WEATHER_STATIONS,
    WeatherStation,
    get_hourly_forecast,
    get_forecast_high_for_date,
)
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("weather_forecast_client")


@dataclass
class ForecastSource:
    """A single forecast from one provider."""
    provider: str              # "nws", "open_meteo_gfs", "open_meteo_ecmwf"
    temperature_high_f: float
    model_name: str


@dataclass
class MultiSourceForecast:
    """Aggregated forecasts from multiple providers for one city/date."""
    city: str
    target_date: str
    sources: List[ForecastSource] = field(default_factory=list)
    failed_sources: List[str] = field(default_factory=list)


# In-memory cache: key -> (timestamp, MultiSourceForecast)
_multi_cache: Dict[str, Tuple[float, MultiSourceForecast]] = {}
_CACHE_TTL = 600  # 10 minutes


def _make_ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def _fetch_nws(station: WeatherStation, target_date: str) -> Optional[ForecastSource]:
    """Fetch high temperature from NWS via existing client."""
    try:
        periods = await get_hourly_forecast(station)
        if not periods:
            return None
        high = get_forecast_high_for_date(periods, target_date)
        if high is None:
            return None
        return ForecastSource(
            provider="nws",
            temperature_high_f=float(high),
            model_name="NWS GFS/blend",
        )
    except Exception as e:
        logger.warning(f"NWS fetch failed for {station.city}: {e}")
        return None


async def _fetch_open_meteo(
    session: aiohttp.ClientSession,
    station: WeatherStation,
    target_date: str,
    model: str,
    provider_name: str,
) -> Optional[ForecastSource]:
    """Fetch high temperature from Open-Meteo for a given model."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={station.lat}&longitude={station.lon}"
        f"&daily=temperature_2m_max"
        f"&models={model}"
        f"&temperature_unit=fahrenheit"
        f"&forecast_days=2"
    )
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=15),
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"Open-Meteo ({model}) returned {resp.status}")
                return None
            data = await resp.json()

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])

        for i, d in enumerate(dates):
            if d == target_date and i < len(highs) and highs[i] is not None:
                return ForecastSource(
                    provider=provider_name,
                    temperature_high_f=float(highs[i]),
                    model_name=model,
                )
        logger.warning(f"Open-Meteo ({model}): no data for {target_date}")
        return None
    except Exception as e:
        logger.warning(f"Open-Meteo ({model}) fetch failed: {e}")
        return None


async def fetch_all_forecasts(
    station: WeatherStation,
    target_date: str,
) -> MultiSourceForecast:
    """
    Fetch forecasts from all 3 providers concurrently.

    Returns a MultiSourceForecast with whatever sources succeeded.
    Failed sources are listed in failed_sources.
    """
    cache_key = f"{station.station_id}_{target_date}"
    if cache_key in _multi_cache:
        ts, cached = _multi_cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return cached

    result = MultiSourceForecast(city=station.city, target_date=target_date)

    async with aiohttp.ClientSession() as session:
        nws_task = _fetch_nws(station, target_date)
        gfs_task = _fetch_open_meteo(
            session, station, target_date,
            model="gfs_seamless", provider_name="open_meteo_gfs",
        )
        ecmwf_task = _fetch_open_meteo(
            session, station, target_date,
            model="ecmwf_ifs025", provider_name="open_meteo_ecmwf",
        )

        outcomes = await asyncio.gather(nws_task, gfs_task, ecmwf_task, return_exceptions=True)

    provider_names = ["nws", "open_meteo_gfs", "open_meteo_ecmwf"]
    for name, outcome in zip(provider_names, outcomes):
        if isinstance(outcome, Exception):
            logger.warning(f"Provider {name} raised exception: {outcome}")
            result.failed_sources.append(name)
        elif outcome is None:
            result.failed_sources.append(name)
        else:
            result.sources.append(outcome)

    _multi_cache[cache_key] = (time.time(), result)

    providers_ok = [s.provider for s in result.sources]
    temps = [f"{s.temperature_high_f:.0f}F" for s in result.sources]
    logger.info(
        f"Forecasts for {station.city} ({target_date}): "
        f"{len(result.sources)}/3 sources — {dict(zip(providers_ok, temps))}, "
        f"failed: {result.failed_sources or 'none'}"
    )
    return result
