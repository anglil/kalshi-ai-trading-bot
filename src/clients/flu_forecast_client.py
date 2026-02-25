"""
Multi-Source Flu/Respiratory Forecast Client

Fetches influenza-like illness (ILI) data from 3 CDC-related sources:
  1. Delphi COVIDcast doctor-visits — smoothed adjusted CLI signal (free, no key)
  2. Delphi Epidata FluView — ILI surveillance API (free, no key)
  3. CDC ILINet — weekly ILI surveillance data (POST endpoint, returns ZIP)

All sources are free and require no API keys.
"""

import asyncio
import aiohttp
import csv
import io
import ssl
import time
import zipfile
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


async def _fetch_delphi_covidcast(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    Fetch ILI-like signal from CMU Delphi COVIDcast API.
    Uses doctor-visits/smoothed_adj_cli which is reliably available.
    Data has ~2 week lag, so we look 14-30 days back.
    """
    today = datetime.now()
    # Data has lag — look 14-30 days back
    end_date = (today - timedelta(days=7)).strftime("%Y%m%d")
    start_date = (today - timedelta(days=30)).strftime("%Y%m%d")

    url = (
        f"https://api.delphi.cmu.edu/epidata/covidcast/"
        f"?data_source=doctor-visits"
        f"&signal=smoothed_adj_cli"
        f"&geo_type=nation"
        f"&geo_value=us"
        f"&time_type=day"
        f"&time_values={start_date}:{end_date}"
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
            logger.warning(f"Delphi COVIDcast: no data (result={data.get('result')})")
            return None

        # Get the most recent value
        latest = epidata[-1]
        value = float(latest.get("value", 0))

        if value <= 0:
            return None

        return ForecastSource(
            provider="delphi_covidcast",
            value=value,
            model_name="Delphi COVIDcast CLI",
        )
    except Exception as e:
        logger.warning(f"Delphi COVIDcast fetch failed: {e}")
        return None


async def _fetch_delphi_fluview(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    Fetch ILI data from Delphi Epidata FluView API.
    Uses the fluview endpoint with national-level data.
    """
    today = datetime.now()
    iso_cal = today.isocalendar()
    current_ew = f"{iso_cal[0]}{iso_cal[1]:02d}"
    # Go back ~8 weeks to find recent data
    start_date = today - timedelta(weeks=8)
    start_cal = start_date.isocalendar()
    start_ew = f"{start_cal[0]}{start_cal[1]:02d}"

    url = (
        f"https://api.delphi.cmu.edu/epidata/fluview/"
        f"?regions=nat"
        f"&epiweeks={start_ew}:{current_ew}"
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
                logger.warning(f"Delphi FluView returned {resp.status}")
                return None
            data = await resp.json()

        epidata = data.get("epidata", [])
        if not epidata:
            logger.warning(f"Delphi FluView: no data (result={data.get('result')})")
            return None

        # Get the most recent week
        latest = epidata[-1]
        # wili = weighted ILI percentage
        wili = latest.get("wili")
        if wili is None:
            wili = latest.get("ili")
        if wili is None or float(wili) <= 0:
            return None

        return ForecastSource(
            provider="delphi_fluview",
            value=float(wili),
            model_name="Delphi FluView ILI",
        )
    except Exception as e:
        logger.warning(f"Delphi FluView fetch failed: {e}")
        return None


async def _fetch_cdc_ilinet(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    Fetch CDC ILINet weekly surveillance data.
    The POST endpoint returns a ZIP containing ILINet.csv.
    """
    url = "https://gis.cdc.gov/grasp/flu2/PostPhase02DataDownload"
    try:
        # First get available seasons from init endpoint
        init_url = "https://gis.cdc.gov/grasp/flu2/GetPhase02InitApp?appVersion=Public"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; FluBot/1.0)",
            "Accept": "application/json",
        }
        async with session.get(
            init_url,
            timeout=aiohttp.ClientTimeout(total=15),
            headers=headers,
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"CDC ILINet init returned {resp.status}")
                return None
            init_data = await resp.json()

        seasons = init_data.get("seasons", [])
        if not seasons:
            logger.warning("CDC ILINet: no seasons available")
            return None

        # Use the most recent season (first in list — sorted desc)
        latest_season_id = seasons[0].get("seasonid", 62)

        payload = {
            "AppVersion": "Public",
            "DatasourceDT": [{"ID": 1, "Name": "ILINet"}],
            "RegionTypeId": 3,  # National
            "SubRegionsDT": [{"ID": 0, "Name": ""}],
            "SeasonsDT": [{"ID": latest_season_id, "Name": ""}],
        }

        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=20),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; FluBot/1.0)",
            },
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"CDC ILINet POST returned {resp.status}")
                return None
            zip_bytes = await resp.read()

        # Response is a ZIP file containing ILINet.csv
        try:
            zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        except zipfile.BadZipFile:
            logger.warning("CDC ILINet: response is not a valid ZIP file")
            return None

        csv_name = None
        for name in zf.namelist():
            if "ILINet" in name and name.endswith(".csv"):
                csv_name = name
                break
        if csv_name is None and zf.namelist():
            csv_name = zf.namelist()[0]

        if csv_name is None:
            logger.warning("CDC ILINet: no CSV found in ZIP")
            return None

        csv_text = zf.read(csv_name).decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(csv_text))

        rows = list(reader)
        if not rows:
            logger.warning("CDC ILINet: CSV has no data rows")
            return None

        # Find weighted ILI in last row
        latest = rows[-1]
        weighted_ili = None
        for key in ["%WEIGHTED ILI", "WEIGHTED ILI", "%_WEIGHTED_ILI", "% WEIGHTED ILI"]:
            if key in latest:
                try:
                    weighted_ili = float(latest[key])
                    break
                except (ValueError, TypeError):
                    pass

        if weighted_ili is None or weighted_ili <= 0:
            # Try unweighted ILI
            for key in ["%UNWEIGHTED ILI", "% UNWEIGHTED ILI", "UNWEIGHTED ILI"]:
                if key in latest:
                    try:
                        weighted_ili = float(latest[key])
                        break
                    except (ValueError, TypeError):
                        pass

        if weighted_ili is None or weighted_ili <= 0:
            logger.warning(f"CDC ILINet: could not find ILI value. Keys: {list(latest.keys())[:10]}")
            return None

        return ForecastSource(
            provider="cdc_ilinet",
            value=weighted_ili,
            model_name="CDC ILINet Weekly",
        )
    except Exception as e:
        logger.warning(f"CDC ILINet fetch failed: {e}")
        return None


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
        covidcast_task = _fetch_delphi_covidcast(session)
        fluview_task = _fetch_delphi_fluview(session)
        ilinet_task = _fetch_cdc_ilinet(session)

        outcomes = await asyncio.gather(covidcast_task, fluview_task, ilinet_task, return_exceptions=True)

    provider_names = ["delphi_covidcast", "delphi_fluview", "cdc_ilinet"]
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
