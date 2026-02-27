# Kalshi AI Trading Bot — Deep Diagnostic Report

**Analysis Date:** February 27, 2026  
**Period Covered:** February 22–27, 2026 (5 full days of live trading)  
**Starting Deposit:** $400.00  
**Current Account Value:** $379.82 (Cash: $75.21 + Portfolio: $304.61)  
**Total P&L:** -$20.18 (-5.0%)

---

## Executive Summary

The bot has been live-trading for 5 days with a $400 deposit. It has executed **626 fills** across **498 orders** in **69+ markets**, settling 85 positions with a **34.1% overall win rate** — well below the ~50% breakeven threshold required for binary options. The account is down **$20.18 (-5.0%)** from the initial deposit, with **$304.61 still locked in 37 open positions**. The primary driver of losses is the **Weather strategy**, which accounts for 93% of all trading activity but achieves only a 33% win rate. The bot exhibits a severe **buy-only bias** — 559 buys vs. only 23 sells in weather alone — meaning it almost never exits positions before settlement, forfeiting any opportunity to cut losses or take profits.

---

## 1. Account Overview

| Metric | Value |
|---|---|
| Starting Deposit | $400.00 |
| Current Cash | $75.21 |
| Current Portfolio Value | $304.61 |
| Current Total | $379.82 |
| Realized P&L (Settlements) | -$20.18 |
| Total Fills | 626 |
| Total Orders | 498 |
| Open Positions | 37 |
| Settled Positions | 85 |
| Taker Rate | 72.2% |
| Total Capital Deployed (cumulative buys) | $1,459.75 |

The bot has deployed **$1,459.75** in cumulative buy orders against a $400 deposit, meaning capital has been recycled approximately 3.6 times through settlements. This high turnover is driven by the weather strategy's daily settlement cycle.

---

## 2. Strategy Breakdown

### 2.1 Fills by Strategy

| Strategy | Fills | Buys | Sells | Buy Volume | Sell Volume | Net Cash Flow | Markets |
|---|---|---|---|---|---|---|---|
| **Weather** | 582 | 559 | 23 | $1,293.90 | $82.57 | -$1,211.33 | 69 |
| Economics | 24 | 20 | 4 | $17.40 | $39.10 | +$21.70 | 8 |
| NBA | 6 | 6 | 0 | $19.70 | $0.00 | -$19.70 | 6 |
| Golf | 5 | 5 | 0 | $42.13 | $0.00 | -$42.13 | 5 |
| Esports | 4 | 4 | 0 | $42.79 | $0.00 | -$42.79 | 4 |
| Other | 5 | 5 | 0 | $43.83 | $0.00 | -$43.83 | 5 |

**Weather dominates everything** — 93% of all fills, 89% of all capital deployed. The other strategies are barely active.

### 2.2 Settlement Results (Realized P&L)

| Strategy | Won | Lost | Win Rate | Revenue | Estimated Cost | Est. Net P&L |
|---|---|---|---|---|---|---|
| **Weather** | 22 | 44 | **33.3%** | $998.00 | ~$1,050 | ~-$52 |
| Golf | 4 | 9 | 30.8% | $42.00 | ~$50 | ~-$8 |
| NBA | 2 | 2 | 50.0% | $20.00 | ~$20 | ~$0 |
| Esports | 1 | 0 | 100.0% | $15.00 | ~$10 | ~+$5 |
| Other | 0 | 1 | 0.0% | $0.00 | ~$5 | ~-$5 |
| **Total** | **29** | **56** | **34.1%** | **$1,075** | ~$1,135 | ~**-$60** |

The 34.1% win rate is catastrophically low for binary options where the breakeven win rate depends on the average entry price. At the bot's average entry price of ~$0.02–0.05 per contract, the expected payout on a win is $1.00 per contract, so the breakeven win rate should be around the average entry price (e.g., 5¢ entry → 5% breakeven). However, the bot is buying contracts at a wide range of prices, and many of the "wins" are on cheap contracts while the losses are on more expensive ones.

### 2.3 Weather Strategy — City-Level Breakdown

| City | Cost Basis | Revenue | Est. P&L | Win Rate |
|---|---|---|---|---|
| Austin (AUS) | $430 | $190 | **-$240** | 29% (5W/12L) |
| New York (NY) | $330 | $210 | **-$120** | 38% (6W/10L) |
| Miami (MIA) | $320 | $345 | **+$25** | 24% (4W/15L) |
| Chicago (CHI) | $225 | $255 | **+$30** | 44% (7W/9L) |

Austin is the worst performer with a massive -$240 loss. Miami has the lowest win rate (24%) but still managed a slight profit because its winning contracts paid out more. Chicago is the best performer both by win rate and P&L.

---

## 3. Critical Bugs and Issues Identified

### BUG #1: Extreme Buy-Only Bias (No Active Risk Management)

The bot executed **599 buys** and only **27 sells** across all strategies. The sell-to-buy ratio is **4.5%** — meaning the bot almost never exits a position before settlement. This is the single biggest structural flaw.

**Root Cause:** The position tracking system (`src/jobs/track.py`) has profit-taking and stop-loss logic, but it relies on `place_profit_taking_orders()` and `place_stop_loss_orders()` in `src/jobs/execute.py`. These functions check positions against thresholds but the **weather strategy explicitly skips profit-taking** with the line:

```python
if strategy.startswith('weather'):
    continue
```

This means weather positions (93% of all positions) are **never** evaluated for early exit. They always hold to settlement, even when the market moves against them.

**Impact:** Severe. A position bought at 50¢ that drops to 10¢ will be held to settlement (likely $0 payout) instead of being sold at 10¢ to recover some capital.

### BUG #2: No Position Sizing Relative to Bankroll

The weather strategy sets `weather_bankroll = bankroll` (the full available balance), then uses `max_position_pct=0.05` (5% per bracket). However, this is 5% of the **current cash balance**, not 5% of the total portfolio. As cash depletes, position sizes shrink, but the bot never stops buying even when cash is very low.

**Impact:** The bot deployed $1,459.75 in buys against a $400 deposit over 5 days, meaning it's recycling settlement proceeds immediately into new positions without any reserve.

### BUG #3: Weather Forecast Edge is Not Working (33% Win Rate)

The core thesis of the weather strategy is that NWS forecasts provide an informational edge over Kalshi market prices. The data shows this is **not the case** — the 33% win rate across 66 settled weather positions is significantly below what would be profitable.

**Root Cause Analysis:**

The consensus algorithm uses a Gaussian model centered on the forecast temperature with sigma values of 1.5–4.0°F. The problem is that **Kalshi weather markets are already very efficient** — the market prices already incorporate NWS and other forecast data. The bot is not finding genuine mispricings; it's finding brackets that appear mispriced because the Gaussian model doesn't match the true probability distribution.

Specifically, the bot buys many "Below X" brackets (e.g., "Below 85.5°F in Austin") at cheap prices (2–10¢), expecting them to pay out $1.00. But these brackets are cheap for a reason — the market correctly prices them as unlikely. The bot's Gaussian model overestimates the probability of extreme outcomes.

### BUG #4: Massive Over-Diversification Within Weather

The bot holds positions in **20 weather markets simultaneously** across 4 cities, often buying multiple overlapping brackets for the same city and date. For example, on Feb 26 it held positions in:

- KXHIGHMIA-26FEB26-B80.5, B78.5, B76.5, T76 (4 Miami brackets)
- KXHIGHAUS-26FEB26-B86.5, B88.5, B84.5 (3 Austin brackets)
- KXHIGHCHI-26FEB26-B44.5, B42.5, T40 (3 Chicago brackets)

These overlapping brackets are **correlated** — if the temperature in Miami is 82°F, then B80.5, B78.5, and B76.5 ALL lose. The bot treats them as independent bets, but they're essentially the same bet with slight variations.

**Impact:** A single temperature outcome wipes out multiple positions simultaneously. The per-city limit (`MAX_POSITIONS_PER_CITY`) exists but is set too high.

### BUG #5: Taker Rate is 72.2% — Paying the Spread

The bot places **72.2% taker orders** (market orders that cross the spread) vs. 27.8% maker orders (limit orders). On Kalshi, taker orders pay higher fees and get worse prices. The bot should be placing more limit orders to capture the spread rather than paying it.

### BUG #6: Follow the Leader Strategy Placed an Esports Bet

The FTL strategy's first live trade was on an esports market (`KXMVESPORTSMULTIGAMEEXTENDED`), which is a very short-duration, high-frequency market — the exact opposite of the "low frequency, span ≥ 1 hour" filter it was supposed to enforce. This suggests the market span filter is not working correctly, possibly because the market metadata doesn't include accurate event duration.

---

## 4. Structural Issues

### 4.1 Night Mode Creates a Timing Disadvantage

The bot pauses most strategies from 11 PM – 7 AM ET. However, weather markets settle based on the **actual high temperature**, which is typically reached between 2–4 PM local time. By the time the bot starts trading at 7 AM ET, the weather markets for that day have already moved significantly based on overnight forecast updates. The bot is buying into markets where early-morning traders have already captured the best prices.

### 4.2 No Exit Strategy for Losing Positions

The bot has no mechanism to sell weather positions that are moving against it. If the morning forecast shifts (e.g., Austin's forecast high drops from 85°F to 80°F), the bot's "Below 85.5°F" position becomes much more valuable, but the bot doesn't sell to lock in the gain. Conversely, if the forecast shifts against the position, the bot holds to settlement and takes a 100% loss.

### 4.3 Capital Efficiency is Poor

With $304.61 locked in 37 open positions and only $75.21 in cash, the bot has **80% of capital deployed** with no ability to take advantage of new opportunities. The bot should maintain a cash reserve of at least 30-40% to handle drawdowns and capitalize on high-conviction signals.

---

## 5. Improvement Recommendations

### Priority 1: Fix the Weather Strategy (Immediate)

| Recommendation | Expected Impact |
|---|---|
| **Reduce max positions per city from current level to 2** | Eliminates correlated bracket losses |
| **Add stop-loss for weather positions** — sell if market price drops below 50% of entry | Recovers capital from losing positions instead of holding to $0 |
| **Increase minimum edge threshold from 8% to 15%** | Filters out marginal signals that aren't truly mispriced |
| **Reduce Kelly fraction from 0.5 to 0.25** | Smaller position sizes reduce impact of losses |
| **Add a daily loss limit per city** — stop trading a city after 2 consecutive losses | Prevents doubling down on bad forecasts |

### Priority 2: Implement Active Position Management

| Recommendation | Expected Impact |
|---|---|
| **Remove the `if strategy.startswith('weather'): continue` skip** in profit-taking | Enables early exit for weather positions |
| **Add a trailing stop-loss** — if a position gains 50%+, set stop at breakeven | Locks in gains instead of risking reversal |
| **Implement time-based exits** — sell positions at 50% of remaining time if underwater | Recovers partial capital instead of waiting for settlement |

### Priority 3: Improve Capital Management

| Recommendation | Expected Impact |
|---|---|
| **Cap total weather exposure at 50% of portfolio** (currently ~80%) | Maintains cash reserve for other strategies |
| **Implement a daily capital deployment limit** — max $100/day in new buys | Prevents over-trading |
| **Increase maker order percentage** — use limit orders with 1-2¢ improvement | Saves on spread costs |

### Priority 4: Diversify Away from Weather

| Recommendation | Expected Impact |
|---|---|
| **Increase FTL strategy allocation** — it follows proven winners | Diversifies into markets with demonstrated edge |
| **Activate economics strategy more aggressively** — it's the only profitable strategy (+$21.70 net) | Capitalize on what's working |
| **Reduce weather to 30% of capital, increase NBA/economics/FTL to 70%** | Better risk distribution |

### Priority 5: Improve the Forecast Model

| Recommendation | Expected Impact |
|---|---|
| **Use ensemble of weather APIs** (not just NWS) — AccuWeather, Weather.com, OpenWeatherMap | Better temperature estimates |
| **Calibrate sigma empirically** — backtest against actual temperature outcomes vs. forecasts | More accurate probability estimates |
| **Add a "market efficiency" check** — if Kalshi price is within 5% of model price, skip | Avoids trading in efficient markets |

---

## 6. Summary of Findings

The bot's -5.0% loss over 5 days is primarily driven by three compounding issues: (1) the weather forecast model does not provide a genuine edge over Kalshi market prices, resulting in a 33% win rate; (2) the bot never exits losing positions early, holding everything to settlement; and (3) correlated bracket positions in the same city amplify losses when a single temperature outcome goes against the forecast. The economics strategy is the only profitable one, but it receives minimal capital allocation. The Follow the Leader strategy has just been activated and needs monitoring. Implementing the Priority 1 and 2 recommendations above should significantly reduce losses and potentially make the bot profitable.

---

*Report generated by deep analysis of 626 fills, 498 orders, 85 settlements, and 37 open positions across 5 days of live trading.*
