"""
NWS (National Weather Service) API Client

Fetches weather forecasts from the free NWS API for use in Kalshi weather
market trading. No API key required — just a User-Agent header.

NWS API docs: https://www.weather.gov/documentation/services-web-api
"""

import asyncio
import aiohttp
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("nws_client")

# Rate limit: NWS asks for max 1 request/second
_last_request_time = 0.0
_NWS_BASE = "https://api.weather.gov"
_USER_AGENT = "(KalshiWeatherBot, contact@example.com)"


@dataclass
class WeatherStation:
    """Maps a Kalshi city to its NWS station and coordinates."""
    city: str
    station_id: str       # NWS station ID (e.g. KNYC)
    lat: float
    lon: float
    kalshi_series: str     # Kalshi series ticker prefix for this city


# Target cities with their NWS stations and Kalshi series tickers
WEATHER_STATIONS: Dict[str, WeatherStation] = {
    "nyc": WeatherStation(
        city="New York City",
        station_id="KNYC",
        lat=40.7128, lon=-73.9352,
        kalshi_series="KXHIGHNY"
    ),
    "chicago": WeatherStation(
        city="Chicago",
        station_id="KMDW",
        lat=41.7868, lon=-87.7522,
        kalshi_series="KXHIGHCHI"
    ),
    "miami": WeatherStation(
        city="Miami",
        station_id="KMIA",
        lat=25.7959, lon=-80.2870,
        kalshi_series="KXHIGHMIA"
    ),
    "austin": WeatherStation(
        city="Austin",
        station_id="KAUS",
        lat=30.1945, lon=-97.6699,
        kalshi_series="KXHIGHAUS"
    ),
    "la": WeatherStation(
        city="Los Angeles",
        station_id="KLAX",
        lat=33.9416, lon=-118.4085,
        kalshi_series="KXHIGHLA"
    ),
}


# Simple in-memory cache
_forecast_cache: Dict[str, Tuple[float, dict]] = {}
_CACHE_TTL = 600  # 10 minutes


async def _rate_limited_get(session: aiohttp.ClientSession, url: str) -> Optional[dict]:
    """Make a rate-limited GET request to the NWS API."""
    global _last_request_time
    
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < 1.1:
        await asyncio.sleep(1.1 - elapsed)
    
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/geo+json"}
    try:
        import ssl
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15), ssl=ssl_ctx) as resp:
            _last_request_time = time.time()
            if resp.status == 200:
                return await resp.json()
            else:
                logger.warning(f"NWS API returned {resp.status} for {url}")
                return None
    except Exception as e:
        logger.error(f"NWS API request failed: {e}")
        return None


async def get_grid_info(session: aiohttp.ClientSession, lat: float, lon: float) -> Optional[str]:
    """Get the NWS grid forecast URL for a lat/lon point."""
    cache_key = f"grid_{lat}_{lon}"
    if cache_key in _forecast_cache:
        ts, data = _forecast_cache[cache_key]
        if time.time() - ts < 86400:  # Grid info doesn't change — cache 24h
            return data
    
    url = f"{_NWS_BASE}/points/{lat},{lon}"
    data = await _rate_limited_get(session, url)
    if data and "properties" in data:
        forecast_url = data["properties"].get("forecastHourly")
        _forecast_cache[cache_key] = (time.time(), forecast_url)
        return forecast_url
    return None


async def get_hourly_forecast(station: WeatherStation) -> Optional[List[Dict]]:
    """
    Fetch hourly forecast for a weather station.
    
    Returns list of hourly periods with:
      - startTime (ISO 8601)
      - temperature (int, °F)
      - temperatureUnit (str)
      - shortForecast (str)
    """
    cache_key = f"hourly_{station.station_id}"
    if cache_key in _forecast_cache:
        ts, data = _forecast_cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return data
    
    try:
        async with aiohttp.ClientSession() as session:
            # Step 1: Get grid forecast URL
            grid_url = await get_grid_info(session, station.lat, station.lon)
            if not grid_url:
                logger.error(f"Could not get grid info for {station.city}")
                return None
            
            # Step 2: Fetch hourly forecast
            data = await _rate_limited_get(session, grid_url)
            if not data or "properties" not in data:
                logger.error(f"Invalid hourly forecast response for {station.city}")
                return None
            
            periods = data["properties"].get("periods", [])
            if not periods:
                logger.warning(f"No forecast periods for {station.city}")
                return None
            
            _forecast_cache[cache_key] = (time.time(), periods)
            logger.info(f"📡 Fetched {len(periods)} hourly forecast periods for {station.city}")
            return periods
            
    except Exception as e:
        logger.error(f"Error fetching hourly forecast for {station.city}: {e}")
        return None


def get_forecast_high_for_date(periods: List[Dict], target_date: str) -> Optional[int]:
    """
    Extract the forecasted daily high temperature for a specific date.
    
    Args:
        periods: Hourly forecast periods from NWS
        target_date: Date string in YYYY-MM-DD format
    
    Returns:
        Forecasted high temperature in °F, or None
    """
    temps_for_date = []
    for period in periods:
        start = period.get("startTime", "")
        if start.startswith(target_date):
            temp = period.get("temperature")
            if temp is not None and period.get("isDaytime", True):
                temps_for_date.append(int(temp))
    
    if temps_for_date:
        return max(temps_for_date)
    return None


def get_forecast_temps_for_date(periods: List[Dict], target_date: str) -> List[int]:
    """
    Get all hourly temperature forecasts for a date (for probability distribution).
    
    Returns list of hourly temperatures during daytime (6 AM - 8 PM).
    """
    temps = []
    for period in periods:
        start = period.get("startTime", "")
        if start.startswith(target_date):
            temp = period.get("temperature")
            if temp is not None:
                # Parse hour from ISO timestamp
                try:
                    dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    hour = dt.hour
                    # Include daytime hours only (6 AM to 8 PM local)
                    if 6 <= hour <= 20:
                        temps.append(int(temp))
                except (ValueError, AttributeError):
                    temps.append(int(temp))  # Include if can't parse time
    return temps


async def get_current_observation(station: WeatherStation) -> Optional[Dict]:
    """
    Get current weather observation for a station.
    
    Returns dict with temperature, conditions, etc.
    """
    cache_key = f"obs_{station.station_id}"
    if cache_key in _forecast_cache:
        ts, data = _forecast_cache[cache_key]
        if time.time() - ts < 300:  # 5 min cache for observations
            return data
    
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{_NWS_BASE}/stations/{station.station_id}/observations/latest"
            data = await _rate_limited_get(session, url)
            if data and "properties" in data:
                props = data["properties"]
                temp_c = props.get("temperature", {}).get("value")
                temp_f = round(temp_c * 9/5 + 32) if temp_c is not None else None
                
                observation = {
                    "temperature_f": temp_f,
                    "description": props.get("textDescription", ""),
                    "timestamp": props.get("timestamp", ""),
                    "wind_speed_mph": props.get("windSpeed", {}).get("value"),
                }
                _forecast_cache[cache_key] = (time.time(), observation)
                return observation
    except Exception as e:
        logger.error(f"Error fetching observation for {station.city}: {e}")
    return None


async def get_all_city_forecasts() -> Dict[str, List[Dict]]:
    """
    Fetch hourly forecasts for all target cities.
    
    Returns dict mapping city key to list of hourly periods.
    """
    results = {}
    for city_key, station in WEATHER_STATIONS.items():
        forecast = await get_hourly_forecast(station)
        if forecast:
            results[city_key] = forecast
        # Small delay between cities to respect rate limits
        await asyncio.sleep(0.5)
    
    logger.info(f"🌡️ Fetched forecasts for {len(results)}/{len(WEATHER_STATIONS)} cities")
    return results
