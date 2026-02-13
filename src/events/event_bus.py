"""
Async publish/subscribe event bus for the Kalshi AI Trading Bot.
Provides decoupled communication between WebSocket streams, position trackers,
and trading strategies via typed events with automatic timestamping.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from src.utils.logging_setup import TradingLoggerMixin


# Canonical event type constants
EVENT_PRICE_UPDATE = "price_update"
EVENT_ORDERBOOK_UPDATE = "orderbook_update"
EVENT_TRADE_EXECUTED = "trade_executed"
EVENT_MARKET_RESOLVED = "market_resolved"
EVENT_POSITION_ALERT = "position_alert"
EVENT_FILL_RECEIVED = "fill_received"

ALL_EVENT_TYPES: Set[str] = {
    EVENT_PRICE_UPDATE,
    EVENT_ORDERBOOK_UPDATE,
    EVENT_TRADE_EXECUTED,
    EVENT_MARKET_RESOLVED,
    EVENT_POSITION_ALERT,
    EVENT_FILL_RECEIVED,
}


@dataclass
class EventData:
    """
    Structured event payload distributed through the event bus.

    Attributes:
        event_type: Canonical event type string (e.g. "price_update").
        data: Arbitrary event payload dict.
        timestamp: UTC datetime when the event was created (auto-populated).
    """
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def matches_ticker(self, ticker: str) -> bool:
        """Check if this event pertains to *ticker*."""
        return self.data.get("ticker") == ticker or self.data.get("market_ticker") == ticker


# Type alias for subscriber callbacks: async callables accepting EventData
Callback = Callable[[EventData], Coroutine[Any, Any, None]]


@dataclass
class _Subscription:
    """Internal representation of a single subscriber."""
    callback: Callback
    ticker_filter: Optional[str] = None  # None means "all tickers"


class EventBus(TradingLoggerMixin):
    """
    Async publish/subscribe event bus (singleton).

    Usage::

        bus = EventBus.get_instance()

        async def on_price(event: EventData):
            print(event.data)

        bus.subscribe("price_update", on_price)
        bus.subscribe("price_update", on_price, ticker="AAPL-24")  # filtered

        await bus.publish("price_update", {"ticker": "AAPL-24", "price": 0.62})

        bus.unsubscribe("price_update", on_price)
    """

    _instance: Optional["EventBus"] = None

    def __init__(self) -> None:
        # event_type -> list of subscriptions
        self._subscribers: Dict[str, List[_Subscription]] = {}
        self._stats: Dict[str, int] = {"published": 0, "delivered": 0, "errors": 0}
        self.logger.info("EventBus initialized")

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls) -> "EventBus":
        """Return the global EventBus singleton, creating it if necessary."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Subscribe / Unsubscribe
    # ------------------------------------------------------------------
    def subscribe(
        self,
        event_type: str,
        callback: Callback,
        ticker: Optional[str] = None,
    ) -> None:
        """
        Register an async callback for *event_type*.

        Args:
            event_type: The event type to listen for.
            callback: An async callable ``(EventData) -> None``.
            ticker: If provided, only events matching this ticker are delivered.
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        # Avoid duplicate registration of the exact same callback + filter
        for sub in self._subscribers[event_type]:
            if sub.callback is callback and sub.ticker_filter == ticker:
                self.logger.debug(
                    "Callback already subscribed",
                    event_type=event_type,
                    ticker=ticker,
                )
                return

        self._subscribers[event_type].append(
            _Subscription(callback=callback, ticker_filter=ticker)
        )
        self.logger.debug(
            "Subscriber registered",
            event_type=event_type,
            ticker_filter=ticker or "all",
        )

    def unsubscribe(
        self,
        event_type: str,
        callback: Callback,
        ticker: Optional[str] = None,
    ) -> None:
        """
        Remove a previously registered callback.

        Args:
            event_type: The event type the callback was registered for.
            callback: The callback to remove.
            ticker: The ticker filter used during subscription (must match).
        """
        subs = self._subscribers.get(event_type, [])
        before = len(subs)
        self._subscribers[event_type] = [
            s for s in subs
            if not (s.callback is callback and s.ticker_filter == ticker)
        ]
        removed = before - len(self._subscribers[event_type])
        if removed:
            self.logger.debug(
                "Subscriber removed",
                event_type=event_type,
                removed_count=removed,
            )
        else:
            self.logger.debug(
                "Subscriber not found for removal",
                event_type=event_type,
            )

    def unsubscribe_all(self, event_type: Optional[str] = None) -> None:
        """
        Remove all subscribers, optionally filtering by *event_type*.
        If *event_type* is ``None`` all subscribers for every event type are removed.
        """
        if event_type is None:
            count = sum(len(v) for v in self._subscribers.values())
            self._subscribers.clear()
            self.logger.info("All subscribers removed", count=count)
        else:
            count = len(self._subscribers.pop(event_type, []))
            self.logger.info(
                "Subscribers removed for event type",
                event_type=event_type,
                count=count,
            )

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------
    async def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Publish an event to all matching subscribers.

        The method creates an :class:`EventData` instance with an automatic UTC
        timestamp and fans the event out to every registered callback whose
        optional ticker filter matches.  If a single subscriber raises an
        exception the remaining subscribers still receive the event.

        Args:
            event_type: Canonical event type string.
            data: Arbitrary payload dict.
        """
        event = EventData(event_type=event_type, data=data)
        self._stats["published"] += 1

        subs = self._subscribers.get(event_type, [])
        if not subs:
            return

        for sub in subs:
            # Apply ticker filter when present
            if sub.ticker_filter is not None and not event.matches_ticker(sub.ticker_filter):
                continue

            try:
                await sub.callback(event)
                self._stats["delivered"] += 1
            except Exception:
                self._stats["errors"] += 1
                self.logger.error(
                    "Subscriber callback failed",
                    event_type=event_type,
                    ticker_filter=sub.ticker_filter,
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def subscriber_count(self, event_type: Optional[str] = None) -> int:
        """Return the number of active subscribers (optionally filtered by type)."""
        if event_type:
            return len(self._subscribers.get(event_type, []))
        return sum(len(v) for v in self._subscribers.values())

    @property
    def stats(self) -> Dict[str, int]:
        """Return a snapshot of publish/deliver/error counters."""
        return dict(self._stats)
