"""
Market Ingestion Job

This job fetches active markets from the Kalshi API, transforms them into a structured format,
and upserts them into the database.
"""
import asyncio
import time
from datetime import datetime
from typing import Optional, List

from src.clients.kalshi_client import KalshiClient
from src.utils.database import DatabaseManager, Market
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger


async def process_and_queue_markets(
    markets_data: List[dict],
    db_manager: DatabaseManager,
    queue: asyncio.Queue,
    existing_position_market_ids: set,
    logger,
):
    """
    Transforms market data, upserts to DB, and puts eligible markets on the queue.
    """
    markets_to_upsert = []
    for market_data in markets_data:
        # A simple approach is to take the average of bid and ask.
        yes_price = (market_data.get("yes_bid", 0) + market_data.get("yes_ask", 0)) / 2
        no_price = (market_data.get("no_bid", 0) + market_data.get("no_ask", 0)) / 2

        volume = int(market_data.get("volume", 0))

        has_position = market_data["ticker"] in existing_position_market_ids

        market = Market(
            market_id=market_data["ticker"],
            title=market_data["title"],
            yes_price=yes_price / 100,
            no_price=no_price / 100,
            volume=volume,
            expiration_ts=int(
                datetime.fromisoformat(
                    market_data["expiration_time"].replace("Z", "+00:00")
                ).timestamp()
            ),
            category=market_data.get("category", "unknown"),
            status=market_data["status"],
            last_updated=datetime.now(),
            has_position=has_position,
        )
        markets_to_upsert.append(market)

    if markets_to_upsert:
        await db_manager.upsert_markets(markets_to_upsert)
        logger.info(f"Successfully upserted {len(markets_to_upsert)} markets.")

        # Primary filtering criteria - MORE PERMISSIVE FOR MORE OPPORTUNITIES!
        min_volume: float = 100.0  # DECREASED: Much lower volume threshold (was 100, keeping low)
        min_volume_for_ai_analysis: float = 150.0  # DECREASED: Lower volume for AI analysis (was 200, now 150)  
        preferred_categories: List[str] = []  # Empty = all categories allowed
        excluded_categories: List[str] = []  # Empty = no categories excluded

        # Enhanced filtering for better opportunities - MORE PERMISSIVE FOR MORE TRADES
        min_price_movement: float = 0.015  # DECREASED: Even lower minimum range (was 0.02, now 1.5¢)
        max_bid_ask_spread: float = 0.20   # INCREASED: Allow even wider spreads (was 0.15, now 20¢)
        min_confidence_for_long_term: float = 0.40  # DECREASED: Lower confidence required (was 0.5, now 40%)

        eligible_markets = [
            m
            for m in markets_to_upsert
            if m.volume >= min_volume
            # REMOVED TIME RESTRICTION - we can now trade markets with ANY deadline!
            # Dynamic exit strategies will handle timing automatically
            and (
                not settings.trading.preferred_categories
                or m.category in settings.trading.preferred_categories
            )
            and m.category not in settings.trading.excluded_categories
        ]

        logger.info(
            f"Found {len(eligible_markets)} eligible markets to process in this batch."
        )
        for market in eligible_markets:
            await queue.put(market)

    else:
        logger.info("No new markets to upsert in this batch.")


async def run_ingestion(
    db_manager: DatabaseManager,
    queue: asyncio.Queue,
    market_ticker: Optional[str] = None,
):
    """
    Main function for the market ingestion job.

    Args:
        db_manager: DatabaseManager instance.
        queue: asyncio.Queue to put ingested markets into.
        market_ticker: Optional specific market ticker to ingest.
    """
    logger = get_trading_logger("market_ingestion")
    logger.info("Starting market ingestion job.", market_ticker=market_ticker)

    kalshi_client = KalshiClient()

    try:
        # Get all market IDs with existing positions
        existing_position_market_ids = await db_manager.get_markets_with_positions()

        if market_ticker:
            logger.info(f"Fetching single market: {market_ticker}")
            market_response = await kalshi_client.get_market(ticker=market_ticker)
            if market_response and "market" in market_response:
                await process_and_queue_markets(
                    [market_response["market"]],
                    db_manager,
                    queue,
                    existing_position_market_ids,
                    logger,
                )
            else:
                logger.warning(f"Could not find market with ticker: {market_ticker}")
        else:
            logger.info("Fetching all active markets from Kalshi API with pagination.")
            cursor = None
            while True:
                response = await kalshi_client.get_markets(limit=100, cursor=cursor)
                markets_page = response.get("markets", [])

                active_markets = [m for m in markets_page if m["status"] == "active"]
                if active_markets:
                    logger.info(
                        f"Fetched {len(markets_page)} markets, {len(active_markets)} are active."
                    )
                    await process_and_queue_markets(
                        active_markets,
                        db_manager,
                        queue,
                        existing_position_market_ids,
                        logger,
                    )

                cursor = response.get("cursor")
                if not cursor:
                    break

    except Exception as e:
        logger.error(
            "An error occurred during market ingestion.", error=str(e), exc_info=True
        )
    finally:
        await kalshi_client.close()
        logger.info("Market ingestion job finished.")
