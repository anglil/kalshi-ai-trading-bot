"""
Multi-Source Flu/Respiratory Forecast Client

Fetches influenza-like illness (ILI) data from 3 CDC-related sources:
  1. CDC FluSight — ensemble forecast (GitHub CSV)
  2. CMU Delphi COVIDcast — ILI signal API (free, no key)
  3. CDC ILINet — weekly ILI surveillance data

All sources are free and require no API keys.
"""

import asyncio
import aiohttp
import csv
import io
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("flu_forecast_client")


# CDC ILI activity level thresholds (% weighted ILI)
ILI_THRESHOLDS = {
    "minimal": (0.0, 2.0),
    "low": (2.0, 2.6),
    "moderate": (2.6, 3.5),
    "high": (3.5, 5.0),
    "very_high": (5.0, 100.0),
}


@dataclass
class ForecastSource:
    """A single flu/ILI forecast from one provider."""
    provider: str
    value: float          # ILI percentage (weighted)
    model_name: str


@dataclass
class MultiSourceForecast:
    """Aggregated flu forecast data from multiple providers."""
    target: str           # e.g. "national_ili"
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


def ili_to_level(ili_pct: float) -> str:
    """Convert ILI percentage to CDC activity level string."""
    for level, (low, high) in ILI_THRESHOLDS.items():
        if low <= ili_pct < high:
            return level
    return "very_high"


def level_to_ili_midpoint(level: str) -> float:
    """Convert a CDC activity level to its midpoint ILI percentage."""
    if level in ILI_THRESHOLDS:
        low, high = ILI_THRESHOLDS[level]
        if high >= 100:
            return low + 1.5  # cap very_high midpoint
        return (low + high) / 2
    return 3.0  # default moderate


async def _fetch_cdc_flusight(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    Fetch CDC FluSight ensemble forecast from GitHub.
    The FluSight challenge publishes ensemble forecasts as CSV on GitHub.
    """
    # FluSight ensemble forecasts are published on cdcgov GitHub
    current_season = _get_flu_season()
    url = (
        f"https://raw.githubusercontent.com/cdcgov/FluSight-forecast-hub/"
        f"main/model-output/FluSight-ensemble/"
    )
    # Try to get the latest forecast file
    index_url = (
        f"https://api.github.com/repos/cdcgov/FluSight-forecast-hub/"
        f"contents/model-output/FluSight-ensemble"
    )
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; FluBot/1.0)",
            "Accept": "application/vnd.github.v3+json",
        }
        async with session.get(
            index_url,
            timeout=aiohttp.ClientTimeout(total=20),
            headers=headers,
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"GitHub FluSight API returned {resp.status}")
                return None
            files = await resp.json()

        # Find most recent CSV file
        csv_files = [f for f in files if isinstance(f, dict) and f.get("name", "").endswith(".csv")]
        if not csv_files:
            logger.warning("FluSight: no CSV files found in repo")
            return None

        csv_files.sort(key=lambda f: f["name"], reverse=True)
        latest_file = csv_files[0]
        download_url = latest_file.get("download_url", "")
        if not download_url:
            return None

        # Download and parse the CSV
        async with session.get(
            download_url,
            timeout=aiohttp.ClientTimeout(total=20),
            headers={"User-Agent": "Mozilla/5.0 (compatible; FluBot/1.0)"},
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                return None
            csv_text = await resp.text()

        # Parse CSV for national ILI estimate
        reader = csv.DictReader(io.StringIO(csv_text))
        national_values = []
        for row in reader:
            location = row.get("location", "")
            output_type = row.get("output_type", "")
            # Look for national (US) median/point forecast
            if location == "US" and output_type in ("median", "point", "quantile"):
                if output_type == "quantile" and row.get("output_type_id", "") != "0.5":
                    continue
                try:
                    val = float(row.get("value", 0))
                    if val > 0:
                        national_values.append(val)
                except (ValueError, TypeError):
                    pass

        if not national_values:
            logger.warning("FluSight: no national ILI data found in CSV")
            return None

        # Use the median value
        national_values.sort()
        mid = len(national_values) // 2
        ili_value = national_values[mid]

        return ForecastSource(
            provider="cdc_flusight",
            value=ili_value,
            model_name="CDC FluSight Ensemble",
        )
    except Exception as e:
        logger.warning(f"CDC FluSight fetch failed: {e}")
        return None


async def _fetch_delphi_covidcast(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    Fetch ILI signal from CMU Delphi COVIDcast API.
    Free API, no key needed.
    """
    # Use the ILI activity indicator from Delphi
    today = datetime.now()
    # Look at the most recent week of data
    end_date = today.strftime("%Y-%m-%d")
    start_date = (today - timedelta(days=14)).strftime("%Y-%m-%d")

    url = (
        f"https://api.delphi.cmu.edu/epidata/covidcast/"
        f"?data_source=hhs"
        f"&signal=confirmed_admissions_influenza_1d"
        f"&geo_type=nation"
        f"&geo_value=us"
        f"&time_type=day"
        f"&time_values={start_date.replace('-', '')}:{end_date.replace('-', '')}"
    )
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; FluBot/1.0)",
            "Accept": "application/json",
        }
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=15),
            headers=headers,
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"Delphi COVIDcast returned {resp.status}")
                return None
            data = await resp.json()

        epidata = data.get("epidata", [])
        if not epidata:
            # Try alternative signal: doctor-visits ILI
            alt_url = (
                f"https://api.delphi.cmu.edu/epidata/covidcast/"
                f"?data_source=doctor-visits"
                f"&signal=smoothed_adj_cli"
                f"&geo_type=nation"
                f"&geo_value=us"
                f"&time_type=day"
                f"&time_values={start_date.replace('-', '')}:{end_date.replace('-', '')}"
            )
            async with session.get(
                alt_url,
                timeout=aiohttp.ClientTimeout(total=15),
                headers=headers,
                ssl=_make_ssl_ctx(),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
            epidata = data.get("epidata", [])

        if not epidata:
            logger.warning("Delphi COVIDcast: no ILI data returned")
            return None

        # Get the most recent value
        latest = epidata[-1]
        value = float(latest.get("value", 0))

        if value <= 0:
            return None

        return ForecastSource(
            provider="delphi",
            value=value,
            model_name="CMU Delphi COVIDcast",
        )
    except Exception as e:
        logger.warning(f"Delphi COVIDcast fetch failed: {e}")
        return None


async def _fetch_cdc_ilinet(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    Fetch CDC ILINet weekly surveillance data.
    Uses the CDC WONDER/Fluview API.
    """
    # CDC FluView API endpoint
    url = "https://gis.cdc.gov/grasp/flu2/PostPhase02DataDownload"
    try:
        # FluView requires a POST with specific parameters
        payload = {
            "AppVersion": "Public",
            "DatasourceDT": [{"ID": 1, "Name": "ILINet"}],
            "RegionTypeId": 3,  # National
            "SubRegionsDT": [{"ID": 0, "Name": ""}],
            "SeasonsDT": [{"ID": _get_current_season_id(), "Name": ""}],
        }
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; FluBot/1.0)",
        }
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=20),
            headers=headers,
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                # Try the simpler CSV endpoint
                return await _fetch_cdc_ilinet_csv(session)
            data = await resp.json()

        # Parse ILINet data
        ili_data = data.get("ILINet", [])
        if not ili_data:
            return await _fetch_cdc_ilinet_csv(session)

        # Get the most recent week
        latest = ili_data[-1] if ili_data else None
        if not latest:
            return None

        weighted_ili = float(latest.get("WEIGHTED ILI", latest.get("weighted_ili", 0)))
        if weighted_ili <= 0:
            return None

        return ForecastSource(
            provider="cdc_ilinet",
            value=weighted_ili,
            model_name="CDC ILINet Weekly",
        )
    except Exception as e:
        logger.warning(f"CDC ILINet fetch failed: {e}")
        return await _fetch_cdc_ilinet_csv(session)


async def _fetch_cdc_ilinet_csv(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """Fallback: fetch CDC ILINet data from the CSV download endpoint."""
    url = "https://gis.cdc.gov/grasp/flu2/GetPhase02InitApp?appVersion=Public"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; FluBot/1.0)",
            "Accept": "application/json",
        }
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=20),
            headers=headers,
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"CDC ILINet CSV fallback returned {resp.status}")
                return None
            data = await resp.json()

        # The init app endpoint returns current season data
        ili_net = data.get("ILINet", [])
        if not ili_net:
            return None

        # Most recent week
        latest = ili_net[-1]
        weighted_ili = float(latest.get("WEIGHTED ILI", latest.get("%WEIGHTED ILI", 0)))
        if weighted_ili <= 0:
            return None

        return ForecastSource(
            provider="cdc_ilinet",
            value=weighted_ili,
            model_name="CDC ILINet Weekly",
        )
    except Exception as e:
        logger.warning(f"CDC ILINet CSV fallback failed: {e}")
        return None


def _get_flu_season() -> str:
    """Get current flu season string (e.g. '2025-2026')."""
    now = datetime.now()
    if now.month >= 8:
        return f"{now.year}-{now.year + 1}"
    return f"{now.year - 1}-{now.year}"


def _get_current_season_id() -> int:
    """Get CDC season ID for current flu season."""
    now = datetime.now()
    if now.month >= 8:
        return now.year - 1999 + 49  # approximate mapping
    return now.year - 2000 + 49


async def fetch_flu_forecasts() -> MultiSourceForecast:
    """
    Fetch flu/ILI data from all 3 providers concurrently.
    Returns a MultiSourceForecast with whatever sources succeeded.
    """
    cache_key = "national_flu"
    if cache_key in _cache:
        ts, cached = _cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return cached

    result = MultiSourceForecast(target="national_ili")

    async with aiohttp.ClientSession() as session:
        flusight_task = _fetch_cdc_flusight(session)
        delphi_task = _fetch_delphi_covidcast(session)
        ilinet_task = _fetch_cdc_ilinet(session)

        outcomes = await asyncio.gather(flusight_task, delphi_task, ilinet_task, return_exceptions=True)

    provider_names = ["cdc_flusight", "delphi", "cdc_ilinet"]
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
    values = [f"{s.value:.2f}%" for s in result.sources]
    logger.info(
        f"Flu forecast data: {len(result.sources)}/3 sources — "
        f"{dict(zip(providers_ok, values))}, "
        f"failed: {result.failed_sources or 'none'}"
    )
    return result
