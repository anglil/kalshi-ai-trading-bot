"""
Multi-Source FOMC Rate Decision Forecast Client

Fetches 3 independent sources for consensus on the next FOMC rate decision:

1. FRED Fed Funds Rate — Uses the effective federal funds rate (DFF) and the
   target range (DFEDTARU/DFEDTARL) to determine the current rate, then
   applies a simple probability model based on rate positioning within the band.

2. RateProbability.com API — Free JSON API providing futures-implied FOMC rate
   probabilities derived from Fed Funds Futures. Updated multiple times daily.
   Endpoint: https://rateprobability.com/api/latest

3. Investing.com Fed Rate Monitor — HTML scrape of FOMC rate probabilities
   that mirrors CME FedWatch data. Provides hold/cut/hike percentages.

All sources produce probability distributions over rate outcomes (hold, cut 25bps,
cut 50bps, hike 25bps, etc.) that map directly to Kalshi FOMC bracket markets.
"""

import asyncio
import aiohttp
import io
import json
import re
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("fed_rate_client")


@dataclass
class FedRateOutcome:
    rate_low: float
    rate_high: float
    probability: float
    change_bps: int


@dataclass
class FedRateForecast:
    provider: str
    meeting_date: str
    outcomes: List[FedRateOutcome]
    current_rate_low: float
    current_rate_high: float
    hold_probability: float
    cut_probability: float
    hike_probability: float


@dataclass
class MultiSourceFedForecast:
    meeting_date: str = ""
    sources: List[FedRateForecast] = field(default_factory=list)
    failed_sources: List[str] = field(default_factory=list)


_cache: Dict[str, Tuple[float, MultiSourceFedForecast]] = {}
_CACHE_TTL = 1800


def _make_ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


FOMC_MEETING_DATES = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16",
    "2027-01-27", "2027-03-17", "2027-05-05", "2027-06-16",
]


def get_next_fomc_meeting() -> Optional[str]:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    for date in FOMC_MEETING_DATES:
        if date >= today:
            return date
    return None


def get_days_to_next_meeting() -> int:
    next_meeting = get_next_fomc_meeting()
    if not next_meeting:
        return 999
    meeting_dt = datetime.strptime(next_meeting, "%Y-%m-%d")
    return (meeting_dt - datetime.utcnow()).days


async def _fetch_fred_fed_rate(
    session: aiohttp.ClientSession,
) -> Optional[FedRateForecast]:
    """Compute FOMC rate probabilities from FRED effective fed funds rate data."""
    next_meeting = get_next_fomc_meeting()
    if not next_meeting:
        return None

    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; FedBot/1.0)", "Accept": "text/csv,*/*"}

        dff_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF"
        async with session.get(dff_url, timeout=aiohttp.ClientTimeout(total=40),
                               headers=headers, ssl=_make_ssl_ctx()) as resp:
            if resp.status != 200:
                return None
            text = await resp.text()

        if "<html" in text[:500].lower():
            return None

        lines = text.strip().split("\n")
        if len(lines) < 2:
            return None

        parts = lines[-1].split(",")
        if len(parts) < 2 or parts[1].strip() in (".", ""):
            return None
        current_eff_rate = float(parts[1].strip())

        current_rate_low = round(round(current_eff_rate * 4) / 4 - 0.25, 2)
        current_rate_high = current_rate_low + 0.25

        logger.info(f"FRED DFF: effective rate = {current_eff_rate:.2f}%, "
                     f"target range = {current_rate_low:.2f}-{current_rate_high:.2f}%")

        tar_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFEDTARU"
        async with session.get(
            tar_url,
            timeout=aiohttp.ClientTimeout(total=40),
            headers=headers, ssl=_make_ssl_ctx(),
        ) as resp:
            if resp.status == 200:
                tar_text = await resp.text()
                if "<html" not in tar_text[:500].lower():
                    tar_lines = tar_text.strip().split("\n")
                    if len(tar_lines) >= 2:
                        tar_parts = tar_lines[-1].split(",")
                        if len(tar_parts) >= 2 and tar_parts[1].strip() not in (".", ""):
                            current_rate_high = float(tar_parts[1].strip())
                            current_rate_low = current_rate_high - 0.25
                            logger.info(f"FRED DFEDTARU: confirmed target = "
                                         f"{current_rate_low:.2f}-{current_rate_high:.2f}%")

        band_position = (current_eff_rate - current_rate_low) / 0.25
        hold_prob = max(0.0, min(1.0, 0.85 + (0.5 - band_position) * 0.1))
        cut_prob = max(0.0, min(1.0, (1.0 - hold_prob) * 0.7))
        hike_prob = max(0.0, 1.0 - hold_prob - cut_prob)

        total = hold_prob + cut_prob + hike_prob
        hold_prob /= total
        cut_prob /= total
        hike_prob /= total

        outcomes = []
        if cut_prob > 0.01:
            outcomes.append(FedRateOutcome(current_rate_low - 0.25, current_rate_high - 0.25, cut_prob, -25))
        outcomes.append(FedRateOutcome(current_rate_low, current_rate_high, hold_prob, 0))
        if hike_prob > 0.01:
            outcomes.append(FedRateOutcome(current_rate_low + 0.25, current_rate_high + 0.25, hike_prob, 25))

        logger.info(f"FRED Fed Rate: hold={hold_prob:.1%}, cut={cut_prob:.1%}, hike={hike_prob:.1%}")

        return FedRateForecast(
            provider="fred_fed_rate", meeting_date=next_meeting, outcomes=outcomes,
            current_rate_low=current_rate_low, current_rate_high=current_rate_high,
            hold_probability=hold_prob, cut_probability=cut_prob, hike_probability=hike_prob,
        )
    except Exception as e:
        logger.warning(f"FRED Fed Rate fetch failed: {e}")
        return None


async def _fetch_rateprobability(
    session: aiohttp.ClientSession,
) -> Optional[FedRateForecast]:
    """Fetch FOMC rate probabilities from rateprobability.com free JSON API."""
    next_meeting = get_next_fomc_meeting()
    if not next_meeting:
        return None

    try:
        url = "https://rateprobability.com/api/latest"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Referer": "https://rateprobability.com/fed",
        }

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20),
                               headers=headers, ssl=_make_ssl_ctx()) as resp:
            if resp.status != 200:
                logger.warning(f"RateProbability API returned {resp.status}")
                return None
            data = await resp.json()

        today = data.get("today", {})
        if not today:
            logger.warning("RateProbability: no 'today' data")
            return None

        band_str = today.get("current band", "")
        midpoint = today.get("midpoint")

        current_rate_low = 3.50
        current_rate_high = 3.75
        if band_str:
            band_match = re.match(r'(\d+\.?\d*)\s*[-\u2013]\s*(\d+\.?\d*)', band_str)
            if band_match:
                current_rate_low = float(band_match.group(1))
                current_rate_high = float(band_match.group(2))

        rows = today.get("rows", [])
        target_row = None
        for row in rows:
            if row.get("meeting_iso") == next_meeting:
                target_row = row
                break

        if not target_row:
            for row in rows:
                if row.get("meeting_iso", "") >= datetime.utcnow().strftime("%Y-%m-%d"):
                    target_row = row
                    break

        if not target_row:
            logger.warning(f"RateProbability: no data for meeting {next_meeting}")
            return None

        prob_move_pct = target_row.get("prob_move_pct", 0)
        prob_is_cut = target_row.get("prob_is_cut", True)
        implied_rate = target_row.get("implied_rate_post_meeting", midpoint)
        change_bps = target_row.get("change_bps", 0)

        move_prob = prob_move_pct / 100.0 if prob_move_pct else 0.0
        hold_prob = 1.0 - move_prob

        if prob_is_cut:
            cut_prob = move_prob
            hike_prob = 0.0
        else:
            cut_prob = 0.0
            hike_prob = move_prob

        outcomes = []
        if cut_prob > 0.01:
            outcomes.append(FedRateOutcome(current_rate_low - 0.25, current_rate_high - 0.25, cut_prob, -25))
        outcomes.append(FedRateOutcome(current_rate_low, current_rate_high, hold_prob, 0))
        if hike_prob > 0.01:
            outcomes.append(FedRateOutcome(current_rate_low + 0.25, current_rate_high + 0.25, hike_prob, 25))

        meeting_label = target_row.get("meeting", next_meeting)
        logger.info(
            f"RateProbability ({meeting_label}): hold={hold_prob:.1%}, cut={cut_prob:.1%}, "
            f"hike={hike_prob:.1%}, implied={implied_rate:.3f}%, change={change_bps}bps"
        )

        return FedRateForecast(
            provider="rateprobability", meeting_date=next_meeting, outcomes=outcomes,
            current_rate_low=current_rate_low, current_rate_high=current_rate_high,
            hold_probability=hold_prob, cut_probability=cut_prob, hike_probability=hike_prob,
        )
    except Exception as e:
        logger.warning(f"RateProbability API fetch failed: {e}")
        return None


async def _fetch_investing_fed_monitor(
    session: aiohttp.ClientSession,
) -> Optional[FedRateForecast]:
    """Scrape FOMC rate probabilities from Investing.com Fed Rate Monitor."""
    next_meeting = get_next_fomc_meeting()
    if not next_meeting:
        return None

    try:
        url = "https://www.investing.com/central-banks/fed-rate-monitor"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0",
            "Accept": "text/html,application/xhtml+xml,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",  # Avoid brotli
        }

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=25),
                               headers=headers, ssl=_make_ssl_ctx()) as resp:
            if resp.status != 200:
                logger.warning(f"Investing.com Fed Monitor returned {resp.status}")
                return None
            html = await resp.text()

        outcomes = []
        current_rate_low = 3.50
        current_rate_high = 3.75

        current_rate_match = re.search(r'current.*?(\d+\.?\d*)\s*[-\u2013]\s*(\d+\.?\d*)%', html, re.IGNORECASE)
        if current_rate_match:
            current_rate_low = float(current_rate_match.group(1))
            current_rate_high = float(current_rate_match.group(2))

        hold_match = re.search(r'(?:hold|no\s*change|maintain|unchanged).*?(\d+\.?\d+)\s*%', html, re.IGNORECASE)
        cut25_match = re.search(r'(?:cut|decrease|lower|ease).*?25.*?(\d+\.?\d+)\s*%', html, re.IGNORECASE)
        cut50_match = re.search(r'(?:cut|decrease|lower|ease).*?50.*?(\d+\.?\d+)\s*%', html, re.IGNORECASE)
        hike25_match = re.search(r'(?:hike|increase|raise|tighten).*?25.*?(\d+\.?\d+)\s*%', html, re.IGNORECASE)

        hold_prob = float(hold_match.group(1)) / 100 if hold_match else 0.0
        cut_prob = 0.0
        hike_prob = 0.0
        if cut25_match:
            cut_prob += float(cut25_match.group(1)) / 100
        if cut50_match:
            cut_prob += float(cut50_match.group(1)) / 100
        if hike25_match:
            hike_prob += float(hike25_match.group(1)) / 100

        if hold_prob > 0 or cut_prob > 0 or hike_prob > 0:
            total = hold_prob + cut_prob + hike_prob
            if total > 0:
                hold_prob /= total
                cut_prob /= total
                hike_prob /= total

            if cut_prob > 0.01:
                outcomes.append(FedRateOutcome(current_rate_low - 0.25, current_rate_high - 0.25, cut_prob, -25))
            outcomes.append(FedRateOutcome(current_rate_low, current_rate_high, hold_prob, 0))
            if hike_prob > 0.01:
                outcomes.append(FedRateOutcome(current_rate_low + 0.25, current_rate_high + 0.25, hike_prob, 25))

            logger.info(f"Investing.com Fed Monitor: hold={hold_prob:.1%}, cut={cut_prob:.1%}, hike={hike_prob:.1%}")

            return FedRateForecast(
                provider="investing_fed_monitor", meeting_date=next_meeting, outcomes=outcomes,
                current_rate_low=current_rate_low, current_rate_high=current_rate_high,
                hold_probability=hold_prob, cut_probability=cut_prob, hike_probability=hike_prob,
            )

        # Fallback: look for any percentage values
        all_probs = re.findall(r'(\d+\.?\d+)\s*%', html)
        if all_probs:
            probs = [float(p) for p in all_probs if 0 < float(p) < 100]
            if probs:
                max_prob = max(probs) / 100
                hold_prob = max_prob
                cut_prob = (1 - hold_prob) * 0.7
                hike_prob = 1 - hold_prob - cut_prob

                outcomes.append(FedRateOutcome(current_rate_low, current_rate_high, hold_prob, 0))
                if cut_prob > 0.01:
                    outcomes.append(FedRateOutcome(current_rate_low - 0.25, current_rate_high - 0.25, cut_prob, -25))

                logger.info(f"Investing.com (fallback): hold={hold_prob:.1%}, cut={cut_prob:.1%}")
                return FedRateForecast(
                    provider="investing_fed_monitor", meeting_date=next_meeting, outcomes=outcomes,
                    current_rate_low=current_rate_low, current_rate_high=current_rate_high,
                    hold_probability=hold_prob, cut_probability=cut_prob, hike_probability=hike_prob,
                )

        logger.warning("Investing.com Fed Monitor: could not parse any probabilities")
        return None
    except Exception as e:
        logger.warning(f"Investing.com Fed Monitor fetch failed: {e}")
        return None


async def fetch_fed_rate_forecasts() -> MultiSourceFedForecast:
    """Fetch FOMC rate decision forecasts from 3 independent sources."""
    next_meeting = get_next_fomc_meeting()
    cache_key = f"fed_rate_{next_meeting}"

    if cache_key in _cache:
        ts, cached = _cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return cached

    result = MultiSourceFedForecast(meeting_date=next_meeting or "")

    async with aiohttp.ClientSession() as session:
        tasks = [
            _fetch_fred_fed_rate(session),
            _fetch_rateprobability(session),
            _fetch_investing_fed_monitor(session),
        ]
        provider_names = ["fred_fed_rate", "rateprobability", "investing_fed_monitor"]
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
    hold_probs = [f"{s.hold_probability:.1%}" for s in result.sources]
    logger.info(
        f"Fed rate forecasts: {len(result.sources)}/3 sources -- "
        f"providers={providers_ok}, hold_probs={hold_probs}, "
        f"failed: {result.failed_sources or 'none'}"
    )
    return result
