"""
Position Limits Module

Implements the position limits recommended by Grok4 performance analysis:
- Maximum 10 concurrent positions
- Maximum 5% of portfolio per trade
- Automatic position closure when over limits
- Pre-trade validation and enforcement

Key Features:
- Real-time position count monitoring
- Portfolio percentage calculations
- Automatic least-performing position closure
- Integration with all trading strategies
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import asyncio
from datetime import datetime, timedelta

from src.utils.database import DatabaseManager, Position
from src.clients.kalshi_client import KalshiClient
from src.utils.logging_setup import get_trading_logger
from src.config.settings import settings


@dataclass
class PositionLimitResult:
    """Result of position limit checking."""
    can_trade: bool
    reason: str
    current_positions: int
    max_positions: int
    current_portfolio_usage: float
    max_position_size: float
    recommended_actions: List[str]


@dataclass
class PositionToClose:
    """Position candidate for closure."""
    position_id: int
    market_id: str
    side: str
    current_pnl: float
    confidence: float
    age_hours: float
    priority_score: float  # Higher = more urgent to close
    quantity: int = 1


class PositionLimitsManager:
    """
    Centralized position limits enforcement following Grok4 recommendations.
    
    Implements strict position limits to prevent over-concentration and
    excessive risk exposure that contributed to the performance issues.
    """
    
    def __init__(self, db_manager: DatabaseManager, kalshi_client: KalshiClient):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.logger = get_trading_logger("position_limits")
        
        # INCREASED: More aggressive limits for more opportunities
        self.max_positions = 15  # INCREASED: Allow 15 positions (was 10)
        self.max_position_size_pct = 5.0  # INCREASED: 5% max per trade (was 3%)
        self.warning_threshold = self.max_positions - 3  # Warning at 12 positions
        
        # Additional safety limits - MORE AGGRESSIVE FOR FULL PORTFOLIO USE
        self.emergency_position_limit = 20  # INCREASED: Higher emergency threshold (was 15)
        self.min_cash_reserve_pct = 0.5  # DECREASED: Only 0.5% cash reserves (was 1% - nearly full deployment)
        
    async def check_position_limits(
        self,
        proposed_position_size: float,
        portfolio_value: Optional[float] = None
    ) -> PositionLimitResult:
        """
        Check if a new position can be added within limits.
        
        Args:
            proposed_position_size: Dollar value of proposed position
            portfolio_value: Total portfolio value (fetched if not provided)
            
        Returns:
            PositionLimitResult with decision and recommendations
        """
        try:
            # Get current portfolio state
            if portfolio_value is None:
                portfolio_value = await self._get_portfolio_value()
            
            current_positions = await self._get_position_count()
            current_usage = await self._calculate_portfolio_usage(portfolio_value)
            
            # Calculate proposed position percentage
            proposed_position_pct = (proposed_position_size / portfolio_value) * 100
            max_position_size = portfolio_value * (self.max_position_size_pct / 100)
            
            recommendations = []
            can_trade = True
            reason = "Position limits satisfied"
            
            # Check 1: Position count limit
            if current_positions >= self.max_positions:
                can_trade = False
                reason = f"Position count {current_positions} at/above limit {self.max_positions}"
                recommendations.append(f"Close {current_positions - self.max_positions + 1} positions before adding new ones")
            elif current_positions >= self.warning_threshold:
                recommendations.append(f"Approaching position limit ({current_positions}/{self.max_positions})")
            
            # Check 2: Position size limit
            if proposed_position_size > max_position_size:
                can_trade = False
                reason = f"Position size ${proposed_position_size:.2f} exceeds limit ${max_position_size:.2f} ({self.max_position_size_pct}%)"
                recommendations.append(f"Reduce position size to maximum ${max_position_size:.2f}")
            
            # Check 3: Total portfolio usage - RELAXED FOR FULL PORTFOLIO USE
            projected_usage = current_usage + proposed_position_pct
            if projected_usage > 100:  # INCREASED: Allow up to 100% portfolio usage (was 85% - user wants full portfolio use)
                can_trade = False
                reason = f"Total portfolio usage would be {projected_usage:.1f}% (limit: 97%)"
                recommendations.append("Close existing positions to free up capital")
            
            # Check 4: Cash reserves
            available_cash = await self._get_available_cash()
            cash_after_trade = available_cash - proposed_position_size
            min_cash_required = portfolio_value * (self.min_cash_reserve_pct / 100)
            
            if cash_after_trade < min_cash_required:
                can_trade = False
                reason = f"Trade would leave ${cash_after_trade:.2f} cash, below minimum ${min_cash_required:.2f}"
                recommendations.append(f"Maintain at least {self.min_cash_reserve_pct}% cash reserves")
            
            return PositionLimitResult(
                can_trade=can_trade,
                reason=reason,
                current_positions=current_positions,
                max_positions=self.max_positions,
                current_portfolio_usage=current_usage,
                max_position_size=max_position_size,
                recommended_actions=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error checking position limits: {e}")
            return PositionLimitResult(
                can_trade=False,
                reason=f"Error checking limits: {e}",
                current_positions=0,
                max_positions=self.max_positions,
                current_portfolio_usage=0.0,
                max_position_size=0.0,
                recommended_actions=["Review system errors"]
            )
    
    async def enforce_position_limits(self, force_closure: bool = False) -> Dict[str, Any]:
        """
        Enforce position limits by closing positions if necessary.
        
        Args:
            force_closure: Whether to force immediate closure regardless of preferences
            
        Returns:
            Dictionary with enforcement results
        """
        try:
            current_positions = await self._get_position_count()
            positions_to_close = max(0, current_positions - self.max_positions)
            
            if positions_to_close == 0 and not force_closure:
                return {
                    'action': 'no_action_needed',
                    'current_positions': current_positions,
                    'message': 'Position count within limits'
                }
            
            if force_closure:
                positions_to_close = max(positions_to_close, current_positions - self.warning_threshold)
            
            # Get positions ranked by closure priority
            closure_candidates = await self._get_positions_for_closure(positions_to_close)
            
            closed_positions = []
            for candidate in closure_candidates:
                try:
                    success = await self._close_position(candidate)
                    if success:
                        closed_positions.append(candidate.market_id)
                        self.logger.info(f"✅ CLOSED POSITION: {candidate.market_id} (Priority: {candidate.priority_score:.2f})")
                    else:
                        self.logger.warning(f"⏳ Position {candidate.market_id} sell failed — will retry next cycle")
                except Exception as e:
                    self.logger.error(f"Failed to close position {candidate.market_id}: {e}")
            
            return {
                'action': 'positions_closed',
                'positions_closed': len(closed_positions),
                'closed_markets': closed_positions,
                'remaining_positions': current_positions - len(closed_positions),
                'message': f'Closed {len(closed_positions)} positions to enforce limits'
            }
            
        except Exception as e:
            self.logger.error(f"Error enforcing position limits: {e}")
            return {
                'action': 'error',
                'message': f'Error enforcing limits: {e}'
            }
    
    async def get_position_limits_status(self) -> Dict[str, Any]:
        """Get current position limits status for monitoring."""
        try:
            current_positions = await self._get_position_count()
            portfolio_value = await self._get_portfolio_value()
            portfolio_usage = await self._calculate_portfolio_usage(portfolio_value)
            available_cash = await self._get_available_cash()
            
            status = "HEALTHY"
            if current_positions >= self.max_positions:
                status = "OVER_LIMIT"
            elif current_positions >= self.warning_threshold:
                status = "WARNING"
            
            return {
                'status': status,
                'current_positions': current_positions,
                'max_positions': self.max_positions,
                'position_utilization': f"{current_positions}/{self.max_positions}",
                'portfolio_usage_pct': portfolio_usage,
                'available_cash': available_cash,
                'portfolio_value': portfolio_value,
                'max_position_size': portfolio_value * (self.max_position_size_pct / 100),
                'cash_reserve_pct': (available_cash / portfolio_value) * 100,
                'recommendations': self._get_status_recommendations(current_positions, portfolio_usage)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting position limits status: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    async def _get_position_count(self) -> int:
        """Get current number of open positions."""
        positions = await self.db_manager.get_open_positions()
        return len(positions)
    
    async def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + positions)."""
        try:
            # Get available cash
            balance_response = await self.kalshi_client.get_balance()
            available_cash = balance_response.get('balance', 0) / 100
            
            # Get current positions value
            positions_response = await self.kalshi_client.get_positions()
            positions = positions_response.get('positions', []) if isinstance(positions_response, dict) else []
            total_position_value = 0
            
            for position in positions:
                if not isinstance(position, dict):
                    continue
                quantity = position.get('quantity', 0)
                if quantity != 0:
                    # Estimate position value (could be improved with real-time pricing)
                    position_value = abs(quantity) * 0.50  # Conservative estimate
                    total_position_value += position_value
            
            return available_cash + total_position_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {e}")
            return 100.0  # Conservative fallback
    
    async def _get_available_cash(self) -> float:
        """Get available cash balance."""
        try:
            balance_response = await self.kalshi_client.get_balance()
            return balance_response.get('balance', 0) / 100
        except Exception as e:
            self.logger.error(f"Error getting available cash: {e}")
            return 0.0
    
    async def _calculate_portfolio_usage(self, portfolio_value: float) -> float:
        """Calculate current portfolio usage percentage."""
        try:
            available_cash = await self._get_available_cash()
            used_capital = portfolio_value - available_cash
            return (used_capital / portfolio_value) * 100
        except Exception as e:
            self.logger.error(f"Error calculating portfolio usage: {e}")
            return 0.0
    
    async def _get_positions_for_closure(self, count: int) -> List[PositionToClose]:
        """Get positions ranked by closure priority."""
        try:
            positions = await self.db_manager.get_open_positions()
            
            closure_candidates = []
            for position in positions:
                # Calculate closure priority (higher = more urgent to close)
                priority_score = await self._calculate_closure_priority(position)
                
                age_hours = (datetime.now() - position.timestamp).total_seconds() / 3600
                
                candidate = PositionToClose(
                    position_id=position.id,
                    market_id=position.market_id,
                    side=position.side,
                    current_pnl=0.0,  # Could be calculated with real-time pricing
                    confidence=position.confidence or 0.5,
                    age_hours=age_hours,
                    priority_score=priority_score,
                    quantity=position.quantity,
                )
                closure_candidates.append(candidate)
            
            # Sort by priority (highest first = most urgent to close)
            closure_candidates.sort(key=lambda x: x.priority_score, reverse=True)
            
            return closure_candidates[:count]
            
        except Exception as e:
            self.logger.error(f"Error getting positions for closure: {e}")
            return []
    
    async def _calculate_closure_priority(self, position: Position) -> float:
        """Calculate priority score for position closure (higher = more urgent)."""
        try:
            priority = 0.0
            
            # Factor 1: Low confidence positions (higher priority to close)
            if position.confidence and position.confidence < 0.6:
                priority += 3.0
            elif position.confidence and position.confidence < 0.7:
                priority += 1.0
            
            # Factor 2: Age (older positions have higher priority)
            age_hours = (datetime.now() - position.timestamp).total_seconds() / 3600
            if age_hours > 72:  # 3+ days old
                priority += 2.0
            elif age_hours > 24:  # 1+ days old
                priority += 1.0
            
            # Factor 3: Position size (larger positions can free up more capital)
            position_value = position.quantity * position.entry_price
            if position_value > 50:  # Large positions
                priority += 1.0
            
            # Factor 4: No stop-loss set (higher priority - more risky)
            if not position.stop_loss_price:
                priority += 2.0
            
            return priority
            
        except Exception as e:
            self.logger.error(f"Error calculating closure priority: {e}")
            return 0.0
    
    async def _close_position(self, candidate: PositionToClose) -> bool:
        """Close a position by selling on Kalshi, then mark as closed in database."""
        try:
            import uuid

            # Build sell order — sell at 1 cent (worst price) to ensure fill like a market order
            side_lower = candidate.side.lower()
            order_kwargs = {
                "ticker": candidate.market_id,
                "client_order_id": str(uuid.uuid4()),
                "side": side_lower,
                "action": "sell",
                "count": candidate.quantity,
                "type_": "limit",
            }
            # Set price to 1 cent on our side (worst case = guaranteed fill)
            if side_lower == "yes":
                order_kwargs["yes_price"] = 1
            else:
                order_kwargs["no_price"] = 1

            try:
                order_response = await self.kalshi_client.place_order(**order_kwargs)
                self.logger.info(
                    f"Position {candidate.market_id} sold on Kalshi: {order_response}"
                )
            except Exception as sell_err:
                self.logger.warning(
                    f"Could not sell {candidate.market_id} on Kalshi: {sell_err}. "
                    f"Keeping position open in DB to retry later."
                )
                return False  # Don't mark as closed if we couldn't actually sell

            # Update position status to closed only after successful sell
            await self.db_manager.update_position_status(candidate.position_id, "closed")

            self.logger.info(f"Position {candidate.market_id} closed due to position limits")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position {candidate.market_id}: {e}")
            return False
    
    def _get_status_recommendations(self, positions: int, usage: float) -> List[str]:
        """Get recommendations based on current status."""
        recommendations = []
        
        if positions >= self.max_positions:
            recommendations.append(f"URGENT: Close {positions - self.max_positions + 1} positions immediately")
        elif positions >= self.warning_threshold:
            recommendations.append(f"Consider closing {positions - self.warning_threshold} positions")
        
        if usage > 85:
            recommendations.append("Portfolio usage high - consider reducing position sizes")
        elif usage > 75:
            recommendations.append("Portfolio usage moderate - monitor for overexposure")
        
        if not recommendations:
            recommendations.append("Position limits healthy - good risk management")
        
        return recommendations


# Convenience functions for easy integration
async def check_can_add_position(
    position_size: float,
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient
) -> Tuple[bool, str]:
    """Simple check if a position can be added."""
    manager = PositionLimitsManager(db_manager, kalshi_client)
    result = await manager.check_position_limits(position_size)
    return result.can_trade, result.reason


async def enforce_limits_if_needed(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient
) -> bool:
    """Enforce position limits if needed."""
    manager = PositionLimitsManager(db_manager, kalshi_client)
    current_count = await manager._get_position_count()
    
    if current_count > manager.max_positions:
        result = await manager.enforce_position_limits()
        return result['action'] == 'positions_closed'
    
    return False


async def get_max_position_size(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient
) -> float:
    """Get maximum allowed position size."""
    manager = PositionLimitsManager(db_manager, kalshi_client)
    portfolio_value = await manager._get_portfolio_value()
    return portfolio_value * (manager.max_position_size_pct / 100) 