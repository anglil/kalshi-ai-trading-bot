"""
Kalshi API client for trading operations.
Handles authentication, market data, and trade execution.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlencode

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin


class KalshiAPIError(Exception):
    """Custom exception for Kalshi API errors."""
    pass


class KalshiClient(TradingLoggerMixin):
    """
    Kalshi API client for automated trading.
    Handles authentication, market data retrieval, and trade execution.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        private_key_path: str = "kalshi_private_key",
        max_retries: int = 5,
        backoff_factor: float = 0.5
    ):
        """
        Initialize Kalshi client.
        
        Args:
            api_key: Kalshi API key (Key ID from the API key generation)
            private_key_path: Path to private key file
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Factor for exponential backoff
        """
        self.api_key = api_key or settings.api.kalshi_api_key
        self.base_url = settings.api.kalshi_base_url
        self.private_key_path = private_key_path
        self.private_key = None
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        # Load private key
        self._load_private_key()
        
        # HTTP client with timeouts
        self.client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )
        
        self.logger.info("Kalshi client initialized", api_key_length=len(self.api_key) if self.api_key else 0)
    
    def _load_private_key(self) -> None:
        """Load private key from file, env var, or committed .b64 fallback."""
        import os
        import base64

        # Build ordered list of key sources to try
        key_sources = []

        # 1. Env-var / constructor path
        primary = os.environ.get('KALSHI_PRIVATE_KEY_PATH', self.private_key_path)
        key_sources.append(('primary', Path(primary)))

        # 2. Committed base64 file (works in GitHub Actions even if secret is wrong)
        b64_fallback = Path(primary).parent / 'kalshi_private_key.b64'
        key_sources.append(('b64_fallback', b64_fallback))

        # 3. Also check repo root if running from a subdirectory
        repo_root_b64 = Path('kalshi_private_key.b64')
        if repo_root_b64.resolve() != b64_fallback.resolve():
            key_sources.append(('repo_root_b64', repo_root_b64))

        last_error = None
        for source_name, key_path in key_sources:
            if not key_path.exists():
                continue
            try:
                raw = key_path.read_bytes()

                # If the file does not start with a PEM header, assume base64
                if not raw.strip().startswith(b'-----'):
                    raw = base64.b64decode(raw)

                self.private_key = serialization.load_pem_private_key(
                    raw,
                    password=None
                )
                self.logger.info("Private key loaded successfully", source=source_name)
                return
            except Exception as e:
                last_error = e
                self.logger.warning(f"Key source '{source_name}' failed: {e}")
                continue

        msg = f"All key sources exhausted. Last error: {last_error}"
        self.logger.error(msg)
        raise KalshiAPIError(f"Failed to load private key: {msg}")
    
    def _sign_request(self, timestamp: str, method: str, path: str) -> str:
        """
        Sign request using RSA PSS signing method as per Kalshi API docs.
        
        Args:
            timestamp: Request timestamp in milliseconds
            method: HTTP method
            path: Request path
        
        Returns:
            Base64 encoded signature
        """
        # Create message to sign: timestamp + method + path
        message = timestamp + method.upper() + path
        message_bytes = message.encode('utf-8')
        
        try:
            # Sign using RSA PSS as per Kalshi documentation
            signature = self.private_key.sign(
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH
                ),
                hashes.SHA256()
            )
            
            return base64.b64encode(signature).decode('utf-8')
        except Exception as e:
            self.logger.error("Failed to sign request", error=str(e))
            raise KalshiAPIError(f"Failed to sign request: {e}")
    
    async def _make_authenticated_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        require_auth: bool = True
    ) -> Dict[str, Any]:
        """
        Make authenticated request to Kalshi API with retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            json_data: JSON request body
            require_auth: Whether authentication is required
        
        Returns:
            API response data
        """
        # Prepare request
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Add authentication headers if required
        if require_auth:
            # Get current timestamp in milliseconds
            timestamp = str(int(time.time() * 1000))
            
            # Create signature
            signature = self._sign_request(timestamp, method, endpoint)
            
            headers.update({
                "KALSHI-ACCESS-KEY": self.api_key,
                "KALSHI-ACCESS-TIMESTAMP": timestamp,
                "KALSHI-ACCESS-SIGNATURE": signature
            })
        
        # Prepare body
        body = None
        if json_data:
            body = json.dumps(json_data, separators=(',', ':'))
        
        # Add query parameters to URL if present
        if params:
            query_string = urlencode(params)
            url = f"{url}?{query_string}"
        
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(
                    "Making API request",
                    method=method,
                    endpoint=endpoint,
                    has_auth=require_auth,
                    attempt=attempt + 1
                )
                
                # Add aggressive delay between requests to prevent 429s
                await asyncio.sleep(0.5)  # 500ms delay = max 2 requests/second
                
                response = await self.client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    content=body if body else None
                )
                
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                last_exception = e
                # Rate limit (429) or server errors (5xx) are worth retrying
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    self.logger.warning(
                        f"API request failed with status {e.response.status_code}. Retrying in {sleep_time:.2f}s...",
                        endpoint=endpoint,
                        attempt=attempt + 1
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    # Don't retry on other client errors (e.g., 400, 401, 404)
                    error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
                    self.logger.error("API request failed without retry", error=error_msg, endpoint=endpoint)
                    raise KalshiAPIError(error_msg)
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Request failed with general exception. Retrying...", error=str(e), endpoint=endpoint)
                sleep_time = self.backoff_factor * (2 ** attempt)
                await asyncio.sleep(sleep_time)
        
        raise KalshiAPIError(f"API request failed after {self.max_retries} retries: {last_exception}")
    
    async def get_balance(self) -> Dict[str, Any]:
        """Get account balance."""
        return await self._make_authenticated_request("GET", "/trade-api/v2/portfolio/balance")
    
    async def get_positions(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """Get portfolio positions."""
        params = {}
        if ticker:
            params["ticker"] = ticker
        return await self._make_authenticated_request("GET", "/trade-api/v2/portfolio/positions", params=params)
    
    async def get_fills(self, ticker: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        """Get order fills.""" 
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        return await self._make_authenticated_request("GET", "/trade-api/v2/portfolio/fills", params=params)
    
    async def get_orders(self, ticker: Optional[str] = None, status: Optional[str] = None) -> Dict[str, Any]:
        """Get orders."""
        params = {}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        return await self._make_authenticated_request("GET", "/trade-api/v2/portfolio/orders", params=params)
    
    async def get_markets(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None,
        tickers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get markets data.

        Args:
            limit: Maximum number of markets to return
            cursor: Pagination cursor
            event_ticker: Filter by event ticker
            series_ticker: Filter by series ticker
            status: Filter by market status (valid: "open", "closed", "settled")
            tickers: List of specific tickers to fetch

        Returns:
            Markets data
        """
        VALID_STATUSES = {"open", "closed", "settled"}
        params = {"limit": limit}

        if cursor:
            params["cursor"] = cursor
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            if status not in VALID_STATUSES:
                self.logger.warning(
                    f"Invalid market status filter '{status}', using 'open' instead. "
                    f"Valid values: {VALID_STATUSES}"
                )
                status = "open"
            params["status"] = status
        if tickers:
            params["tickers"] = ",".join(tickers)
        
        return await self._make_authenticated_request(
            "GET", "/trade-api/v2/markets", params=params, require_auth=True
        )
    
    async def get_market(self, ticker: str) -> Dict[str, Any]:
        """Get specific market data."""
        return await self._make_authenticated_request(
            "GET", f"/trade-api/v2/markets/{ticker}", require_auth=False
        )
    
    async def get_orderbook(self, ticker: str, depth: int = 100) -> Dict[str, Any]:
        """
        Get market orderbook.
        
        Args:
            ticker: Market ticker
            depth: Orderbook depth
        
        Returns:
            Orderbook data
        """
        params = {"depth": depth}
        return await self._make_authenticated_request(
            "GET", f"/trade-api/v2/markets/{ticker}/orderbook", params=params, require_auth=False
        )
    
    async def get_market_history(
        self,
        ticker: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get market price history.
        
        Args:
            ticker: Market ticker
            start_ts: Start timestamp
            end_ts: End timestamp
            limit: Number of records to return
        
        Returns:
            Price history data
        """
        params = {"limit": limit}
        if start_ts:
            params["start_ts"] = start_ts
        if end_ts:
            params["end_ts"] = end_ts
        
        return await self._make_authenticated_request(
            "GET", f"/trade-api/v2/markets/{ticker}/history", params=params, require_auth=False
        )
    
    async def place_order(
        self,
        ticker: str,
        client_order_id: str,
        side: str,
        action: str,
        count: int,
        type_: str = "market",
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        expiration_ts: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Place a trading order.
        
        Args:
            ticker: Market ticker
            client_order_id: Unique client order ID
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            type_: Order type ("market" or "limit")
            yes_price: Yes price in cents (for limit orders)
            no_price: No price in cents (for limit orders)
            expiration_ts: Order expiration timestamp
        
        Returns:
            Order response
        """
        order_data = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": side,
            "action": action,
            "count": count,
            "type": type_
        }
        
        if yes_price is not None:
            order_data["yes_price"] = yes_price
        if no_price is not None:
            order_data["no_price"] = no_price
        if expiration_ts:
            order_data["expiration_ts"] = expiration_ts
        
        return await self._make_authenticated_request(
            "POST", "/trade-api/v2/portfolio/orders", json_data=order_data
        )
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        return await self._make_authenticated_request(
            "DELETE", f"/trade-api/v2/portfolio/orders/{order_id}"
        )

    async def get_total_portfolio_value(self) -> float:
        """
        Get total portfolio value (cash + position market values) in dollars.
        This provides the true bankroll for position sizing.
        """
        try:
            balance_response = await self.get_balance()
            cash = balance_response.get("balance", 0) / 100.0

            positions_response = await self.get_positions()
            position_value = 0.0
            for pos in positions_response.get("market_positions", []):
                qty = abs(pos.get("position", 0))
                if qty == 0:
                    continue
                market_price = pos.get("market_price", 50)  # cents
                position_value += qty * market_price / 100.0

            return cash + position_value
        except Exception:
            # Fallback to cash-only if positions API fails
            try:
                balance_response = await self.get_balance()
                return balance_response.get("balance", 0) / 100.0
            except Exception:
                return 0.0
    
    async def get_trades(
        self,
        ticker: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get trade history.
        
        Args:
            ticker: Filter by ticker
            limit: Maximum number of trades to return
            cursor: Pagination cursor
        
        Returns:
            Trades data
        """
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor
        
        return await self._make_authenticated_request(
            "GET", "/trade-api/v2/portfolio/trades", params=params
        )
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
        self.logger.info("Kalshi client closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close() 