"""
Multi-Source Economic Indicator Forecast Client

Fetches 3 independent sources per indicator for majority-vote consensus:

CPI (3 sources, all YoY %):
  1. BLS API v1 — CPI-U NSA (CUUR0000SA0), no key needed
  2. FRED CSV  — CPI-U SA (CPIAUCSL), no key needed
  3. FRED CSV  — Core CPI SA (CPILFESL, less food & energy), no key needed

NFP (3 sources, all MoM change in K):
  1. BLS API v1 — Total Nonfarm (CES0000000001), no key needed
  2. FRED CSV  — Total Nonfarm mirror (PAYEMS), no key needed
  3. FRED CSV  — CPS Household Employment (CE16OV, independent survey), no key needed

GDP (3 sources, all Q/Q annualized %):
  1. FRED CSV — Atlanta Fed GDPNow nowcast (GDPNOW), no key needed
  2. FRED CSV — Official GDP growth rate (A191RL1Q225SBEA), no key needed
  3. FRED CSV — Gross Domestic Income growth (A261RL1Q225SBEA, independent), no key needed

BLS v1 API: free, no key, 25 requests/day limit.
FRED CSV: free, no key, no rate limit concerns.
"""

import asyncio
import aiohttp
import csv
import io
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


# Indicator definitions — tolerances set for 3-source majority voting
ECON_INDICATORS = {
    "CPI": EconIndicator(
        name="CPI",
        kalshi_series_candidates=["CPI", "KXCPI", "INFLATION", "CPIYOY"],
        unit="%",
        cluster_tolerance=0.3,   # NSA vs SA vs Core typically differ by 0.1-0.25%
        sigma_high=0.05,
        sigma_med=0.1,
        sigma_low=0.2,
    ),
    "NFP": EconIndicator(
        name="NFP",
        kalshi_series_candidates=["NFP", "KXNFP", "JOBS", "NONFARM", "PAYROLLS"],
        unit="K",
        cluster_tolerance=50.0,  # BLS+PAYEMS always match; CPS is volatile
        sigma_high=15.0,
        sigma_med=30.0,
        sigma_low=60.0,
    ),
    "GDP": EconIndicator(
        name="GDP",
        kalshi_series_candidates=["GDP", "KXGDP", "GDPGROWTH"],
        unit="%",
        cluster_tolerance=1.5,   # nowcast vs official vs GDI can differ ~1%
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


# ---------------------------------------------------------------------------
# FRED CSV helper — reused by many sources
# ---------------------------------------------------------------------------

async def _fetch_fred_csv(
    session: aiohttp.ClientSession,
    series_id: str,
    label: str,
) -> Optional[List[Dict[str, str]]]:
    """
    Fetch a FRED CSV and return parsed rows as list of dicts.
    Columns are DATE and the series_id.
    Returns None on failure.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; EconBot/1.0)",
            "Accept": "text/csv,*/*",
        }
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=20),
            headers=headers,
            ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"FRED {label} ({series_id}) returned {resp.status}")
                return None
            text = await resp.text()

        # Detect Cloudflare HTML blocks
        if "<html" in text[:500].lower():
            logger.warning(f"FRED {label} ({series_id}): got HTML instead of CSV")
            return None

        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            logger.warning(f"FRED {label} ({series_id}): no data rows")
            return None
        return rows
    except Exception as e:
        logger.warning(f"FRED {label} ({series_id}) fetch failed: {e}")
        return None


def _fred_yoy(rows: List[Dict[str, str]], series_id: str) -> Optional[float]:
    """Compute YoY % change from FRED monthly index data (last 13 months)."""
    valid = [r for r in rows
             if r.get(series_id, ".") not in (".", "", "-")]
    if len(valid) < 13:
        return None
    cur = float(valid[-1][series_id])
    ago = float(valid[-13][series_id])
    if ago <= 0:
        return None
    return (cur - ago) / ago * 100


def _fred_mom(rows: List[Dict[str, str]], series_id: str) -> Optional[float]:
    """Compute MoM change from FRED monthly level data (last 2 months)."""
    valid = [r for r in rows
             if r.get(series_id, ".") not in (".", "", "-")]
    if len(valid) < 2:
        return None
    cur = float(valid[-1][series_id])
    prev = float(valid[-2][series_id])
    return cur - prev


# ===========================================================================
# CPI sources (3)
# ===========================================================================

async def _fetch_bls_cpi(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    CPI source 1: BLS API v1, series CUUR0000SA0 (CPI-U, Not Seasonally Adjusted).
    Computes year-over-year percentage change.
    """
    url = "https://api.bls.gov/publicAPI/v1/timeseries/data/CUUR0000SA0"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; EconBot/1.0)",
            "Accept": "application/json",
        }
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=20),
            headers=headers, ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"BLS CPI API returned {resp.status}")
                return None
            data = await resp.json()

        if data.get("status") != "REQUEST_SUCCEEDED":
            logger.warning(f"BLS CPI: {data.get('status')}")
            return None

        series_data = data.get("Results", {}).get("series", [])
        if not series_data:
            return None

        all_data = series_data[0].get("data", [])
        monthly = [d for d in all_data if d.get("period", "").startswith("M")
                    and d.get("period") != "M13"
                    and d.get("value", "-") != "-"]

        if len(monthly) < 13:
            return None

        current_value = float(monthly[0]["value"])
        year_ago_value = float(monthly[12]["value"])
        if year_ago_value <= 0:
            return None

        yoy_pct = (current_value - year_ago_value) / year_ago_value * 100
        logger.info(f"BLS CPI-U NSA: {current_value} / {year_ago_value} = {yoy_pct:.2f}% YoY")

        return ForecastSource(
            provider="bls_cpi",
            value=round(yoy_pct, 2),
            model_name="BLS CPI-U NSA YoY",
            indicator="CPI",
        )
    except Exception as e:
        logger.warning(f"BLS CPI fetch failed: {e}")
        return None


async def _fetch_fred_cpi(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    CPI source 2: FRED CPIAUCSL (CPI-U All Items, Seasonally Adjusted).
    Independent seasonal adjustment gives slightly different YoY than NSA.
    """
    rows = await _fetch_fred_csv(session, "CPIAUCSL", "CPI-U SA")
    if not rows:
        return None
    yoy = _fred_yoy(rows, "CPIAUCSL")
    if yoy is None:
        return None
    logger.info(f"FRED CPI-U SA: YoY = {yoy:.2f}%")
    return ForecastSource(
        provider="fred_cpi",
        value=round(yoy, 2),
        model_name="FRED CPI-U SA YoY",
        indicator="CPI",
    )


async def _fetch_fred_core_cpi(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    CPI source 3: FRED CPILFESL (Core CPI — Less Food and Energy, SA).
    Independent basket excluding volatile food & energy; usually within 0.3%
    of headline CPI.
    """
    rows = await _fetch_fred_csv(session, "CPILFESL", "Core CPI")
    if not rows:
        return None
    yoy = _fred_yoy(rows, "CPILFESL")
    if yoy is None:
        return None
    logger.info(f"FRED Core CPI: YoY = {yoy:.2f}%")
    return ForecastSource(
        provider="fred_core_cpi",
        value=round(yoy, 2),
        model_name="FRED Core CPI YoY",
        indicator="CPI",
    )


# ===========================================================================
# NFP sources (3)
# ===========================================================================

async def _fetch_bls_nfp(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    NFP source 1: BLS API v1, series CES0000000001 (Total Nonfarm, SA).
    Computes month-over-month change in thousands.
    """
    url = "https://api.bls.gov/publicAPI/v1/timeseries/data/CES0000000001"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; EconBot/1.0)",
            "Accept": "application/json",
        }
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=20),
            headers=headers, ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"BLS NFP API returned {resp.status}")
                return None
            data = await resp.json()

        if data.get("status") != "REQUEST_SUCCEEDED":
            return None

        series_data = data.get("Results", {}).get("series", [])
        if not series_data:
            return None

        all_data = series_data[0].get("data", [])
        monthly = [d for d in all_data if d.get("period", "").startswith("M")
                    and d.get("period") != "M13"]
        if len(monthly) < 2:
            return None

        current_value = float(monthly[0]["value"])
        prev_value = float(monthly[1]["value"])
        mom_change_k = current_value - prev_value

        logger.info(f"BLS NFP: {current_value}K - {prev_value}K = {mom_change_k:+.0f}K MoM")

        return ForecastSource(
            provider="bls_nfp",
            value=round(mom_change_k, 0),
            model_name="BLS Total Nonfarm MoM",
            indicator="NFP",
        )
    except Exception as e:
        logger.warning(f"BLS NFP fetch failed: {e}")
        return None


async def _fetch_fred_nfp(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    NFP source 2: FRED PAYEMS (Total Nonfarm Payrolls, SA).
    Same BLS data via FRED CSV — serves as cross-verification path.
    """
    rows = await _fetch_fred_csv(session, "PAYEMS", "PAYEMS")
    if not rows:
        return None
    mom = _fred_mom(rows, "PAYEMS")
    if mom is None:
        return None
    logger.info(f"FRED PAYEMS: MoM = {mom:+.0f}K")
    return ForecastSource(
        provider="fred_nfp",
        value=round(mom, 0),
        model_name="FRED Total Nonfarm MoM",
        indicator="NFP",
    )


async def _fetch_fred_household(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    NFP source 3: FRED CE16OV (CPS Civilian Employment Level, SA).
    Independent household survey (vs employer survey for NFP).
    More volatile but genuinely independent — when it agrees with CES,
    confidence is high.
    """
    rows = await _fetch_fred_csv(session, "CE16OV", "CPS Employment")
    if not rows:
        return None
    mom = _fred_mom(rows, "CE16OV")
    if mom is None:
        return None
    logger.info(f"FRED CE16OV (household): MoM = {mom:+.0f}K")
    return ForecastSource(
        provider="fred_household",
        value=round(mom, 0),
        model_name="CPS Household Employment MoM",
        indicator="NFP",
    )


# ===========================================================================
# GDP sources (3)
# ===========================================================================

async def _fetch_atlanta_fed_gdpnow(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    GDP source 1: FRED GDPNOW (Atlanta Fed GDPNow nowcast).
    Forward-looking real-time GDP estimate, updated throughout the quarter.
    """
    rows = await _fetch_fred_csv(session, "GDPNOW", "GDPNow")
    if not rows:
        return None
    last = rows[-1]
    value_str = last.get("GDPNOW", "")
    if not value_str or value_str == ".":
        return None
    value = float(value_str)
    date = last.get("DATE", "?")
    logger.info(f"FRED GDPNow: {value:.2f}% (date={date})")
    return ForecastSource(
        provider="atlanta_fed",
        value=round(value, 2),
        model_name="Atlanta Fed GDPNow (via FRED)",
        indicator="GDP",
    )


async def _fetch_fred_gdp_official(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    GDP source 2: FRED A191RL1Q225SBEA (Real GDP Growth Rate, Q/Q SAAR).
    Official BEA release — backward-looking but anchors expectations.
    """
    rows = await _fetch_fred_csv(session, "A191RL1Q225SBEA", "GDP Official")
    if not rows:
        return None
    # Get last valid value
    valid = [r for r in rows if r.get("A191RL1Q225SBEA", ".") not in (".", "", "-")]
    if not valid:
        return None
    last = valid[-1]
    value = float(last["A191RL1Q225SBEA"])
    date = last.get("DATE", "?")
    logger.info(f"FRED GDP Official: {value:.1f}% Q/Q SAAR (date={date})")
    return ForecastSource(
        provider="fred_gdp",
        value=round(value, 2),
        model_name="BEA GDP Growth Q/Q SAAR",
        indicator="GDP",
    )


async def _fetch_fred_gdi(session: aiohttp.ClientSession) -> Optional[ForecastSource]:
    """
    GDP source 3: FRED A261RL1Q225SBEA (Real GDI Growth Rate, Q/Q SAAR).
    Gross Domestic Income — independent measure of economic output using
    income data (wages, profits, taxes) rather than expenditure data.
    GDP and GDI should equal in theory but differ due to measurement errors.
    """
    rows = await _fetch_fred_csv(session, "A261RL1Q225SBEA", "GDI")
    if not rows:
        return None
    valid = [r for r in rows if r.get("A261RL1Q225SBEA", ".") not in (".", "", "-")]
    if not valid:
        return None
    last = valid[-1]
    value = float(last["A261RL1Q225SBEA"])
    date = last.get("DATE", "?")
    logger.info(f"FRED GDI: {value:.1f}% Q/Q SAAR (date={date})")
    return ForecastSource(
        provider="fred_gdi",
        value=round(value, 2),
        model_name="BEA GDI Growth Q/Q SAAR",
        indicator="GDP",
    )


# ===========================================================================
# Dispatch — 3 sources per indicator, fetched concurrently
# ===========================================================================

async def fetch_econ_forecasts(indicator: str) -> MultiSourceForecast:
    """
    Fetch economic forecasts for a given indicator from 3 independent sources.

    CPI: BLS CPI-U NSA + FRED CPI-U SA + FRED Core CPI — all YoY %
    NFP: BLS Total Nonfarm + FRED PAYEMS + FRED CPS Household — all MoM K
    GDP: FRED GDPNow + FRED GDP Official + FRED GDI — all Q/Q SAAR %

    3 sources enables majority-vote consensus (2/3 agree = medium, 3/3 = high).
    """
    cache_key = f"econ_{indicator}"
    if cache_key in _cache:
        ts, cached = _cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return cached

    result = MultiSourceForecast(indicator=indicator)

    async with aiohttp.ClientSession() as session:
        if indicator == "CPI":
            tasks = [_fetch_bls_cpi(session), _fetch_fred_cpi(session), _fetch_fred_core_cpi(session)]
            provider_names = ["bls_cpi", "fred_cpi", "fred_core_cpi"]
        elif indicator == "NFP":
            tasks = [_fetch_bls_nfp(session), _fetch_fred_nfp(session), _fetch_fred_household(session)]
            provider_names = ["bls_nfp", "fred_nfp", "fred_household"]
        elif indicator == "GDP":
            tasks = [_fetch_atlanta_fed_gdpnow(session), _fetch_fred_gdp_official(session), _fetch_fred_gdi(session)]
            provider_names = ["atlanta_fed", "fred_gdp", "fred_gdi"]
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
    ind_cfg = ECON_INDICATORS.get(indicator)
    unit = ind_cfg.unit if ind_cfg else ""
    values = [f"{s.value:.2f}{unit}" for s in result.sources]
    logger.info(
        f"Econ forecasts ({indicator}): {len(result.sources)}/3 sources — "
        f"{dict(zip(providers_ok, values))}, "
        f"failed: {result.failed_sources or 'none'}"
    )
    return result
