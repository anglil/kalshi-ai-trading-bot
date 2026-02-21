"""
XAI client for AI-powered trading decisions.
Interfaces with Grok models through xAI SDK for market analysis and trading strategies.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import pickle
import os

import re
from json_repair import repair_json

from openai import AsyncOpenAI

from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin, log_error_with_context
from src.utils.prompts import SIMPLIFIED_PROMPT_TPL


@dataclass
class TradingDecision:
    """Represents an AI trading decision."""
    action: str  # "buy", "sell", "hold"
    side: str    # "yes", "no"
    confidence: float  # 0.0 to 1.0
    limit_price: Optional[int] = None # The limit price for the order in cents.


@dataclass
class DailyUsageTracker:
    """Track daily AI usage and costs."""
    date: str
    total_cost: float = 0.0
    request_count: int = 0
    daily_limit: float = 50.0  # Default $50 daily limit
    is_exhausted: bool = False
    last_exhausted_time: Optional[datetime] = None


class XAIClient(TradingLoggerMixin):
    """
    AI client for trading decisions.
    Now uses Gemini models via Google AI Studio's OpenAI-compatible API.
    """
    
    def __init__(self, api_key: Optional[str] = None, db_manager=None):
        """
        Initialize AI client with Google AI Studio (Gemini) backend.
        
        Args:
            api_key: Gemini API key (defaults to settings)
            db_manager: Optional DatabaseManager for logging queries
        """
        self.api_key = api_key or settings.api.gemini_api_key or settings.api.openrouter_api_key
        self.db_manager = db_manager
        
        # Initialize OpenAI-compatible client pointing at Google AI Studio
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=120.0,
            max_retries=0,  # We handle retries ourselves
        )
        
        # Model configuration
        self.primary_model = settings.trading.primary_model
        self.fallback_model = settings.trading.fallback_model
        self.temperature = settings.trading.ai_temperature
        self.max_tokens = settings.trading.ai_max_tokens
        
        # Cost tracking
        self.total_cost = 0.0
        self.request_count = 0
        
        # Daily usage tracking
        self.daily_tracker = self._load_daily_tracker()
        self.usage_file = "logs/daily_ai_usage.pkl"
        
        # API exhaustion state
        self.is_api_exhausted = False
        self.api_exhausted_until = None
        
        self.logger.info(
            "AI client initialized (Gemini via Google AI Studio)",
            primary_model=self.primary_model,
            logging_enabled=bool(db_manager),
            daily_limit=self.daily_tracker.daily_limit,
            today_cost=self.daily_tracker.total_cost,
            today_requests=self.daily_tracker.request_count
        )

    def _load_daily_tracker(self) -> DailyUsageTracker:
        """Load or create daily usage tracker."""
        today = datetime.now().strftime("%Y-%m-%d")
        usage_file = "logs/daily_ai_usage.pkl"
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        try:
            if os.path.exists(usage_file):
                with open(usage_file, 'rb') as f:
                    tracker = pickle.load(f)
                    
                # Reset if new day
                if tracker.date != today:
                    tracker = DailyUsageTracker(
                        date=today,
                        daily_limit=tracker.daily_limit  # Keep same limit
                    )
                return tracker
        except Exception as e:
            self.logger.warning(f"Failed to load daily tracker: {e}")
        
        # Create new tracker
        daily_limit = getattr(settings.trading, 'daily_ai_cost_limit', 50.0)
        return DailyUsageTracker(date=today, daily_limit=daily_limit)

    def _save_daily_tracker(self):
        """Save daily usage tracker to disk."""
        try:
            os.makedirs("logs", exist_ok=True)
            with open(self.usage_file, 'wb') as f:
                pickle.dump(self.daily_tracker, f)
        except Exception as e:
            self.logger.error(f"Failed to save daily tracker: {e}")

    def _update_daily_cost(self, cost: float):
        """Update daily cost tracking."""
        self.daily_tracker.total_cost += cost
        self.daily_tracker.request_count += 1
        self._save_daily_tracker()
        
        # Check if we've hit daily limit
        if self.daily_tracker.total_cost >= self.daily_tracker.daily_limit:
            self.daily_tracker.is_exhausted = True
            self.daily_tracker.last_exhausted_time = datetime.now()
            self._save_daily_tracker()
            
            self.logger.warning(
                "Daily AI cost limit reached! Trading paused until tomorrow.",
                daily_cost=self.daily_tracker.total_cost,
                daily_limit=self.daily_tracker.daily_limit,
                requests_today=self.daily_tracker.request_count
            )

    async def _check_daily_limits(self) -> bool:
        """
        Check if we should pause due to daily limits.
        Returns True if we can proceed, False if we should sleep.
        """
        # Reload tracker to get latest state
        self.daily_tracker = self._load_daily_tracker()
        
        if self.daily_tracker.is_exhausted:
            now = datetime.now()
            
            # Check if it's a new day
            if self.daily_tracker.date != now.strftime("%Y-%m-%d"):
                # New day - reset tracker
                self.daily_tracker = DailyUsageTracker(
                    date=now.strftime("%Y-%m-%d"),
                    daily_limit=self.daily_tracker.daily_limit
                )
                self._save_daily_tracker()
                
                self.logger.info(
                    "New day - daily AI limits reset",
                    daily_limit=self.daily_tracker.daily_limit
                )
                return True
            
            # Still the same day and exhausted
            self.logger.info(
                "Daily AI limit reached - request skipped",
                daily_cost=self.daily_tracker.total_cost,
                daily_limit=self.daily_tracker.daily_limit
            )
            return False
        
        return True

    async def _handle_resource_exhausted_error(self, error_msg: str):
        """Handle xAI API resource exhausted errors."""
        self.logger.error(
            "xAI API credits exhausted",
            error_details=error_msg,
            daily_cost=self.daily_tracker.total_cost,
            requests_today=self.daily_tracker.request_count
        )
        
        # Mark API as exhausted
        self.is_api_exhausted = True
        self.api_exhausted_until = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + \
                                  timedelta(days=1)  # Next day at midnight
        
        # Also mark daily tracker as exhausted
        self.daily_tracker.is_exhausted = True
        self.daily_tracker.last_exhausted_time = datetime.now()
        self._save_daily_tracker()

    def _is_resource_exhausted_error(self, error: Exception) -> bool:
        """Check if error is due to permanent resource exhaustion (not temp rate limits)."""
        error_str = str(error).lower()
        # Only mark as truly exhausted for billing/credit issues, not per-minute rate limits
        # Gemini free tier returns "quota" errors for per-minute limits — those are temporary
        is_permanent = any(indicator in error_str for indicator in [
            "credits",
            "spending limit",
            "billing",
            "payment required",
        ])
        # Exclude temporary rate limits (which Gemini uses)
        if "quota" in error_str and "retry" in error_str:
            return False  # Temporary rate limit, not permanent exhaustion
        return is_permanent

    async def _log_query(
        self,
        strategy: str,
        query_type: str,
        prompt: str,
        response: str,
        market_id: Optional[str] = None,
        tokens_used: Optional[int] = None,
        cost_usd: Optional[float] = None,
        confidence_extracted: Optional[float] = None,
        decision_extracted: Optional[str] = None
    ):
        """Log an LLM query if database manager is available."""
        if self.db_manager:
            try:
                from src.utils.database import LLMQuery
                
                llm_query = LLMQuery(
                    timestamp=datetime.now(),
                    strategy=strategy,
                    query_type=query_type,
                    market_id=market_id,
                    prompt=prompt[:2000],  # Truncate very long prompts
                    response=response[:5000],  # Truncate very long responses
                    tokens_used=tokens_used,
                    cost_usd=cost_usd,
                    confidence_extracted=confidence_extracted,
                    decision_extracted=decision_extracted
                )
                
                # Use asyncio to avoid blocking
                asyncio.create_task(self.db_manager.log_llm_query(llm_query))
                
            except Exception as e:
                self.logger.error(f"Failed to log LLM query: {e}")

    async def search(self, query: str, max_length: int = 300) -> str:
        """
        Perform a search using Gemini for context generation.
        Uses the AI model to provide relevant context based on its training data.
        """
        try:
            # Process and optimize the search query
            optimized_query = self._optimize_search_query(query)
            
            # Check cache first (simple in-memory cache)
            if hasattr(self, '_search_cache'):
                cache_key = f"{optimized_query[:50]}:{max_length}"
                if cache_key in self._search_cache:
                    self.logger.debug("Returning cached search result", query=optimized_query[:50])
                    return self._search_cache[cache_key]
            else:
                self._search_cache = {}
            
            self.logger.debug(
                "Starting AI-powered search",
                original_query=query[:50],
                optimized_query=optimized_query[:50],
                max_length=max_length
            )
            
            # Use Gemini completion for context generation
            search_prompt = self._create_search_prompt(optimized_query, max_length)
            
            start_time = time.time()
            try:
                response = await self.client.chat.completions.create(
                    model=self.primary_model,
                    messages=[{"role": "user", "content": search_prompt}],
                    temperature=0.3,
                    max_tokens=min(2000, self.max_tokens),
                )
                processing_time = time.time() - start_time
                
                if not response.choices or not response.choices[0].message.content:
                    self.logger.warning(
                        "Search returned empty result",
                        query=optimized_query[:50],
                        processing_time=processing_time
                    )
                    return self._get_fallback_context(query, max_length)
                
                search_result = self._truncate_news_summary(response.choices[0].message.content, max_length)
                search_result += "\n[Based on AI model knowledge]"
                
                # Cache the result
                cache_key = f"{optimized_query[:50]}:{max_length}"
                if len(self._search_cache) < 100:
                    self._search_cache[cache_key] = search_result
                
                self.logger.info(
                    "AI search completed",
                    query=optimized_query[:50],
                    processing_time=round(processing_time, 2)
                )
                
                return search_result
                
            except Exception as sample_error:
                self.logger.warning(
                    "Search failed", 
                    query=optimized_query[:50],
                    error=str(sample_error),
                )
                return self._get_fallback_context(query, max_length)
                
        except Exception as e:
            self.logger.warning(
                "Search failed, using fallback",
                query=query[:50],
                error=str(e),
            )
            return self._get_fallback_context(query, max_length)
    
    def _optimize_search_query(self, query: str) -> str:
        """
        Optimize search queries to improve results and reduce API costs.
        """
        # Remove prediction market specific language that doesn't help with search
        query = query.replace("Will the", "")
        query = query.replace("**", "")  # Remove markdown bold
        query = query.replace("on Jul 18, 2025?", "July 2025")
        query = query.replace("before 2025-", "2025 ")
        
        # Extract key terms for better search results
        if "high temp" in query and ("LA" in query or "Los Angeles" in query):
            return "Los Angeles weather forecast July 2025 temperature"
        elif "high temp" in query and ("Philadelphia" in query or "Philly" in query):
            return "Philadelphia weather forecast July 2025 temperature"
        elif "Rotten Tomatoes" in query:
            # Extract movie name
            movie_part = query.split("Rotten Tomatoes")[0].strip()
            return f"{movie_part} movie Rotten Tomatoes score reviews"
        elif "YoungBoy" in query and "release" in query:
            return "NBA YoungBoy album release date 2025 MASA"
        
        # Generic optimization
        query = query[:150]  # Reasonable length limit
        return query.strip()
    
    def _create_search_prompt(self, query: str, max_length: int) -> str:
        """
        Create an optimized search prompt for better results.
        """
        return f"""Find current, relevant information about: {query}

Focus on:
- Recent news, data, or announcements
- Factual information from reliable sources
- Current conditions or forecasts if applicable

Provide a brief, factual summary under {max_length//2} words. If no current information is available, clearly state that."""
    
    def _process_search_response(self, response: Any, original_query: str, processing_time: float, max_length: int) -> str:
        """
        Process the search response and add metadata.
        """
        # Calculate costs and usage
        sources_used = getattr(response.usage, 'num_sources_used', 0) if hasattr(response, 'usage') else 0
        search_cost = sources_used * 0.025  # $0.025 per source
        
        self.total_cost += search_cost
        self.request_count += 1
        
        self.logger.info(
            "xAI search completed successfully",
            query=original_query[:50],
            sources_used=sources_used,
            search_cost=search_cost,
            processing_time=processing_time,
            has_citations=bool(getattr(response, 'citations', None))
        )
        
        # Truncate response to requested length
        search_result = self._truncate_news_summary(response.content, max_length)
        
        # Add useful metadata
        if sources_used > 0:
            search_result += f"\n[Based on {sources_used} live sources]"
        elif hasattr(response, 'citations') and response.citations:
            citation_count = len(response.citations)
            search_result += f"\n[Based on {citation_count} sources]"
        else:
            search_result += "\n[Based on model knowledge]"
        
        return search_result
    
    def _get_fallback_context(self, query: str, max_length: int) -> str:
        """
        Provide meaningful fallback context when search fails.
        """
        # Provide specific fallbacks based on query type
        if "temp" in query.lower() and ("LA" in query or "Philadelphia" in query):
            city = "Los Angeles" if "LA" in query else "Philadelphia"
            return f"Weather forecasts for {city} in July typically show temperatures varying by day and weather patterns. Precise daily predictions require current meteorological data. [Fallback: Search unavailable]"
        
        elif "Rotten Tomatoes" in query:
            return f"Movie ratings and reviews vary based on critic and audience feedback. Rotten Tomatoes scores depend on the number and sentiment of reviews received. [Fallback: Search unavailable]"
        
        elif "release" in query.lower() and "album" in query.lower():
            return f"Music release dates are typically announced by artists or labels through official channels. Release schedules can change based on various factors. [Fallback: Search unavailable]"
        
        else:
            truncated_query = query[:100] + "..." if len(query) > 100 else query
            return f"Current information about '{truncated_query}' requires live data access. Analyzing based on available market data and general knowledge. [Fallback: Search unavailable]"

    async def get_trading_decision(
        self,
        market_data: Dict,
        portfolio_data: Dict,
        news_summary: str = ""
    ) -> Optional[TradingDecision]:
        """
        Get a trading decision from the AI with improved token management.
        """
        try:
            # First try with full prompt
            decision = await self._get_trading_decision_with_prompt(
                market_data, portfolio_data, news_summary, use_simplified=False
            )
            
            if decision:
                return decision
            
            # If that fails, try with simplified prompt
            self.logger.info("Trying simplified prompt for trading decision")
            decision = await self._get_trading_decision_with_prompt(
                market_data, portfolio_data, news_summary, use_simplified=True
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error getting trading decision: {str(e)}")
            return None

    async def _get_trading_decision_with_prompt(
        self,
        market_data: Dict,
        portfolio_data: Dict,
        news_summary: str = "",
        use_simplified: bool = False
    ) -> Optional[TradingDecision]:
        """
        Get trading decision with either full or simplified prompt.
        """
        try:
            if use_simplified:
                prompt = self._create_simplified_trading_prompt(market_data, portfolio_data, news_summary)
            else:
                prompt = self._create_full_trading_prompt(market_data, portfolio_data, news_summary)
            
            messages = [{"role": "user", "content": prompt}]
            
            # Use appropriate token limits
            max_tokens = 4000 if use_simplified else None  # Use default for full prompt
            
            response_text, cost = await self._make_completion_request(
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens
            )
            
            if not response_text:
                return None
            
            # Parse the JSON response
            return self._parse_trading_decision(response_text)
            
        except Exception as e:
            self.logger.error(f"Error in _get_trading_decision_with_prompt (simplified={use_simplified}): {str(e)}")
            return None

    def _create_simplified_trading_prompt(
        self,
        market_data: Dict,
        portfolio_data: Dict,
        news_summary: str
    ) -> str:
        """
        Create a simplified, token-efficient prompt for trading decisions.
        """
        # Extract key information
        title = market_data.get('title', 'Unknown Market')
        yes_price = (market_data.get('yes_bid', 0) + market_data.get('yes_ask', 100)) / 2
        no_price = (market_data.get('no_bid', 0) + market_data.get('no_ask', 100)) / 2
        volume = market_data.get('volume', 0)
        
        # Truncate news summary to save tokens
        truncated_news = news_summary[:500] + "..." if len(news_summary) > 500 else news_summary
        
        # Create concise prompt
        prompt = f"""Analyze this prediction market and decide whether to trade.

Market: {title}
YES: {yes_price}¢ | NO: {no_price}¢ | Volume: ${volume:,.0f}

News: {truncated_news}

Rules:
- Only trade if you have >10% edge (your probability - market price)
- High confidence (>60%) required
- Return JSON only

Required format:
{{"action": "BUY|SKIP", "side": "YES|NO", "limit_price": 1-99, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""
        
        return prompt

    def _create_full_trading_prompt(
        self,
        market_data: Dict,
        portfolio_data: Dict,
        news_summary: str
    ) -> str:
        """
        Create the full trading prompt with detailed analysis.
        """
        from src.utils.prompts import MULTI_AGENT_PROMPT_TPL
        
        # Use the existing comprehensive prompt
        return MULTI_AGENT_PROMPT_TPL.format(
            title=market_data.get('title', 'Unknown Market'),
            rules=market_data.get('rules', 'No specific rules provided'),
            yes_price=(market_data.get('yes_bid', 0) + market_data.get('yes_ask', 100)) / 2,
            no_price=(market_data.get('no_bid', 0) + market_data.get('no_ask', 100)) / 2,
            volume=market_data.get('volume', 0),
            days_to_expiry=market_data.get('days_to_expiry', 30),
            news_summary=news_summary,
            cash=portfolio_data.get('cash', 1000),
            max_trade_value=portfolio_data.get('max_trade_value', 100),
            max_position_pct=portfolio_data.get('max_position_pct', 5),
            ev_threshold=10  # 10% EV threshold
        )

    def _parse_trading_decision(self, response_text: str) -> Optional[TradingDecision]:
        """
        Parse the trading decision response from the AI.
        """
        try:
            import json
            import re
            
            # Extract JSON from the response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    self.logger.warning("No JSON found in trading decision response")
                    return None
            
            decision_data = json.loads(json_str)
            
            # Normalize the action
            action = decision_data.get('action', 'SKIP').upper()
            if action in ['BUY_YES', 'BUY']:
                action = 'BUY'
            elif action in ['BUY_NO']:
                action = 'BUY'
            elif action in ['SKIP', 'HOLD', 'PASS']:
                action = 'SKIP'
            
            # Extract other fields
            side = decision_data.get('side', 'YES').upper()
            confidence = float(decision_data.get('confidence', 0.5))
            limit_price = int(decision_data.get('limit_price', 50))
            reasoning = decision_data.get('reasoning', 'No reasoning provided')
            
            return TradingDecision(
                action=action,
                side=side,
                confidence=confidence,
                limit_price=limit_price,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing trading decision: {str(e)}")
            self.logger.debug(f"Raw response: {response_text[:500]}")
            return None

    def _prepare_prompt(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any], 
        news_summary: str
    ) -> str:
        """Prepare the prompt for trading decision."""
        max_trade_value = min(
            portfolio_data.get("balance", 0) * settings.trading.max_position_size_pct / 100,
            portfolio_data.get("balance", 0) * 0.05  # 5% max fallback
        )
        
        # Calculate days to expiry
        import datetime
        close_time = market_data.get("close_time", "Unknown")
        days_to_expiry = "Unknown"
        
        if close_time != "Unknown":
            try:
                # Parse close_time and calculate days to expiry
                if isinstance(close_time, str):
                    # Try different datetime formats
                    for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                        try:
                            close_dt = datetime.datetime.strptime(close_time, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        # If no format works, use current time
                        close_dt = datetime.datetime.now()
                elif hasattr(close_time, 'timestamp'):
                    close_dt = close_time
                else:
                    close_dt = datetime.datetime.now()
                
                now = datetime.datetime.now()
                if close_dt.tzinfo is None:
                    close_dt = close_dt.replace(tzinfo=datetime.timezone.utc)
                if now.tzinfo is None:
                    now = now.replace(tzinfo=datetime.timezone.utc)
                    
                delta = close_dt - now
                days_to_expiry = max(0, delta.days)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse close_time: {e}")
                days_to_expiry = 0
        
        prompt_params = {
            "ticker": market_data.get("ticker", "UNKNOWN"),
            "title": market_data.get("title", "Unknown Market"),
            "yes_price": market_data.get("yes_bid", 0),
            "no_price": market_data.get("no_bid", 0),
            "volume": market_data.get("volume", 0),
            "close_time": close_time,
            "days_to_expiry": days_to_expiry,
            "news_summary": news_summary[:1000],  # Limit news length
            "cash": portfolio_data.get("balance", 0),  # Use 'cash' as expected by template
            "balance": portfolio_data.get("balance", 0),
            "existing_positions": len(portfolio_data.get("positions", [])),
            "ev_threshold": settings.trading.min_confidence_to_trade * 100,
            "max_trade_value": max_trade_value,
            "max_position_pct": settings.trading.max_position_size_pct
        }
        
        return SIMPLIFIED_PROMPT_TPL.format(**prompt_params)

    async def _make_completion_request(
        self, 
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3
    ) -> Tuple[Optional[str], float]:
        """
        Make a completion request via Google AI Studio's OpenAI-compatible API.
        """
        # Check daily limits first
        if not await self._check_daily_limits():
            return None, 0.0
        
        # Check if API is exhausted from previous errors
        if self.is_api_exhausted:
            now = datetime.now()
            if self.api_exhausted_until and now < self.api_exhausted_until:
                self.logger.info("API exhausted - skipping request until reset")
                return None, 0.0
            else:
                self.is_api_exhausted = False
                self.api_exhausted_until = None
        
        model_to_use = model or self.primary_model
        temperature = temperature if temperature is not None else self.temperature
        
        from src.config.settings import settings
        if max_tokens is None:
            max_tokens = settings.trading.ai_max_tokens
        
        # Convert xAI SDK message format to OpenAI format if needed
        openai_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                openai_messages.append(msg)
            else:
                # Handle xAI SDK message objects
                openai_messages.append({"role": "user", "content": str(msg)})
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                response = await self.client.chat.completions.create(
                    model=model_to_use,
                    messages=openai_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                processing_time = time.time() - start_time
                
                if not response.choices or not response.choices[0].message.content:
                    self.logger.warning(
                        f"Empty response on attempt {attempt + 1}",
                        model=model_to_use,
                        processing_time=processing_time,
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    # Try fallback
                    if model_to_use == settings.trading.primary_model:
                        fallback_result = await self._try_fallback_model(openai_messages, temperature, max_tokens)
                        if fallback_result:
                            return fallback_result
                    return None, 0.0
                
                response_content = response.choices[0].message.content
                
                # Track cost (near-zero on free tier)
                input_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
                output_tokens = getattr(response.usage, 'completion_tokens', 0) or 0
                cost = (input_tokens + output_tokens) * 0.000001  # Near-zero for free tier
                
                self.total_cost += cost
                self.request_count += 1
                self._update_daily_cost(cost)
                
                self.logger.debug(
                    "Gemini completion successful",
                    model=model_to_use,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                    processing_time=round(processing_time, 2),
                    attempt=attempt + 1
                )
                
                return response_content, cost
                
            except Exception as e:
                if self._is_resource_exhausted_error(e):
                    await self._handle_resource_exhausted_error(str(e))
                    return None, 0.0
                
                is_rate_limit = any(indicator in str(e).lower() for indicator in ["rate limit", "quota", "429", "too many"])
                
                self.logger.warning(
                    f"Error on attempt {attempt + 1}: {str(e)}",
                    model=model_to_use,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    is_rate_limit=is_rate_limit
                )
                
                if attempt == max_retries - 1:
                    if model_to_use == settings.trading.primary_model:
                        self.logger.warning(f"{model_to_use} failed with error, trying fallback model: {str(e)}")
                        fallback_result = await self._try_fallback_model(openai_messages, temperature, max_tokens)
                        if fallback_result:
                            return fallback_result
                    self.logger.error(f"Error in get_completion: {str(e)}")
                    return None, 0.0
                
                delay = 2 ** attempt
                if is_rate_limit:
                    delay *= 2
                await asyncio.sleep(delay)
        
        return None, 0.0

    async def _try_fallback_model(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Optional[Tuple[str, float]]:
        """
        Try using the fallback model when the primary model fails.
        """
        try:
            from src.config.settings import settings
            fallback_model = settings.trading.fallback_model
            
            self.logger.info(f"Attempting fallback to {fallback_model}")
            
            fallback_max_tokens = min(max_tokens or self.max_tokens, 4000)
            
            response = await self.client.chat.completions.create(
                model=fallback_model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=fallback_max_tokens,
            )
            
            if response.choices and response.choices[0].message.content:
                response_content = response.choices[0].message.content
                input_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
                output_tokens = getattr(response.usage, 'completion_tokens', 0) or 0
                cost = (input_tokens + output_tokens) * 0.000001
                
                self.total_cost += cost
                self.request_count += 1
                
                self.logger.info(
                    f"✅ Fallback model {fallback_model} succeeded",
                    response_length=len(response_content),
                    cost=cost
                )
                
                return response_content, cost
            else:
                self.logger.warning(f"Fallback model {fallback_model} returned empty response")
                return None
                
        except Exception as e:
            self.logger.error(f"Fallback model failed: {str(e)}")
            return None

    async def get_completion(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        strategy: str = "unknown",
        query_type: str = "completion",
        market_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Get a simple completion from the AI model.
        Returns the raw response text or None if failed/exhausted.
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            response_content, cost = await self._make_completion_request(
                messages, 
                max_tokens=max_tokens, 
                temperature=temperature
            )
            
            # Check if we got a None response (API exhausted or failed)
            if response_content is None:
                self.logger.info(
                    "AI completion skipped due to API limits or exhaustion",
                    strategy=strategy,
                    query_type=query_type,
                    market_id=market_id
                )
                return None
            
            # Log the query and response
            await self._log_query(
                strategy=strategy,
                query_type=query_type,
                prompt=prompt,
                response=response_content,
                market_id=market_id,
                cost_usd=cost
            )
            
            return response_content
                    
        except Exception as e:
            self.logger.error(f"Error in get_completion: {e}")
            return None

    async def close(self) -> None:
        """No-op for xAI client as it doesn't require closing."""
        self.logger.info(
            "xAI client closed",
            total_estimated_cost=self.total_cost,
            total_requests=self.request_count
        ) 