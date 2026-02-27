"""
Sports Trading Common Utilities

Shared dataclasses, odds conversion, consensus algorithm, signal generation,
and trade execution for NBA and soccer consensus strategies.

Sports markets are direct probability markets (team X wins), so there is no
Gaussian bracket model. Consensus = median of win probabilities from sources.
Edge = consensus probability - Kalshi market price.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from statistics import median
from typing import Dict, List, Optional

from src.clients.kalshi_client import KalshiClient
from src.paper.tracker import log_signal
from src.utils.database import DatabaseManager, Position
from src.utils.logging_setup import get_trading_logger
from src.strategies.weather_strategy import kalshi_taker_fee

logger = get_trading_logger("sports_common")


# ============================================================
# Data structures
# ============================================================

@dataclass
class SportsForecastSource:
    """A single forecast from one provider."""
    provider: str       # "espn_bpi", "elo", "odds_api", etc.
    home_win: float     # probability 0-1
    away_win: float     # probability 0-1
    draw: float = 0.0   # probability 0-1 (0 for NBA)


@dataclass
class SportsMultiSourceForecast:
    """Aggregated forecasts from multiple providers for one game."""
    game_id: str                   # unique game identifier
    home_team: str                 # normalized team name
    away_team: str                 # normalized team name
    league: str                    # "nba", "epl", "ucl", etc.
    sources: List[SportsForecastSource] = field(default_factory=list)
    failed_sources: List[str] = field(default_factory=list)
    start_time: Optional[str] = None  # ISO-8601 game start


@dataclass
class SportsConsensusResult:
    """Result of sports multi-source consensus."""
    home_prob: float        # consensus home win probability
    away_prob: float        # consensus away win probability
    draw_prob: float        # consensus draw probability (0 for NBA)
    confidence: str         # "high", "medium", "low"
    max_spread: float       # max disagreement across sources
    source_count: int


@dataclass
class SportsMarketOutcome:
    """A single Kalshi market outcome for a sports game."""
    ticker: str             # Kalshi market ticker
    outcome: str            # "home", "away", "draw"
    yes_price: int          # current YES price in cents
    no_price: int           # current NO price in cents
    yes_ask: int            # best YES ask in cents
    no_ask: int             # best NO ask in cents
    volume: int = 0

    @property
    def implied_prob(self) -> float:
        """Market implied probability from YES price."""
        return self.yes_price / 100.0


@dataclass
class SportsTradeSignal:
    """A trade signal for a sports market."""
    market: SportsMarketOutcome
    our_prob: float          # our estimated probability
    market_prob: float       # market implied probability
    edge: float              # our_prob - market_prob
    side: str                # "YES" or "NO"
    confidence: float        # how confident we are (0-1)
    limit_price: int         # our limit price in cents
    position_size_dollars: float
    shares: int
    game_desc: str           # e.g. "MIA vs PHI"
    rationale: str


# ============================================================
# Team name normalization
# ============================================================

# Maps common external names to Kalshi ticker codes (3-letter)
NBA_TEAM_MAP: Dict[str, str] = {
    # Full names
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL", "LA Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
    # Short names (ESPN sometimes uses these)
    "Hawks": "ATL", "Celtics": "BOS", "Nets": "BKN", "Hornets": "CHA",
    "Bulls": "CHI", "Cavaliers": "CLE", "Cavs": "CLE",
    "Mavericks": "DAL", "Mavs": "DAL", "Nuggets": "DEN", "Pistons": "DET",
    "Warriors": "GSW", "Rockets": "HOU", "Pacers": "IND",
    "Clippers": "LAC", "Lakers": "LAL", "Grizzlies": "MEM",
    "Heat": "MIA", "Bucks": "MIL", "Timberwolves": "MIN", "Wolves": "MIN",
    "Pelicans": "NOP", "Knicks": "NYK", "Thunder": "OKC",
    "Magic": "ORL", "76ers": "PHI", "Sixers": "PHI",
    "Suns": "PHX", "Trail Blazers": "POR", "Blazers": "POR",
    "Kings": "SAC", "Spurs": "SAS", "Raptors": "TOR",
    "Jazz": "UTA", "Wizards": "WAS",
    # Abbreviations
    "ATL": "ATL", "BOS": "BOS", "BKN": "BKN", "CHA": "CHA", "CHI": "CHI",
    "CLE": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GSW": "GSW",
    "GS": "GSW", "HOU": "HOU", "IND": "IND", "LAC": "LAC", "LAL": "LAL",
    "MEM": "MEM", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NOP": "NOP",
    "NO": "NOP", "NYK": "NYK", "NY": "NYK", "OKC": "OKC", "ORL": "ORL",
    "PHI": "PHI", "PHX": "PHX", "POR": "POR", "SAC": "SAC", "SAS": "SAS",
    "SA": "SAS", "TOR": "TOR", "UTA": "UTA", "WAS": "WAS",
}

SOCCER_TEAM_MAP: Dict[str, str] = {
    # EPL
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU",
    "Brentford": "BRE", "Brighton": "BHA", "Brighton & Hove Albion": "BHA",
    "Chelsea": "CHE", "CFC": "CHE", "Crystal Palace": "CRY",
    "Everton": "EVE", "Fulham": "FUL", "Ipswich": "IPS", "Ipswich Town": "IPS",
    "Leicester": "LEI", "Leicester City": "LEI",
    "Liverpool": "LIV", "Manchester City": "MCI", "Man City": "MCI",
    "Manchester United": "MUN", "Man United": "MUN", "Man Utd": "MUN",
    "Newcastle": "NEW", "Newcastle United": "NEW",
    "Nottingham Forest": "NFO", "Nott'm Forest": "NFO",
    "Southampton": "SOU", "Tottenham": "TOT", "Tottenham Hotspur": "TOT",
    "Spurs": "TOT", "West Ham": "WHU", "West Ham United": "WHU",
    "Wolverhampton": "WOL", "Wolves": "WOL", "Wolverhampton Wanderers": "WOL",
    # La Liga
    "Real Madrid": "RMA", "Barcelona": "BAR", "FC Barcelona": "BAR",
    "Atletico Madrid": "ATM", "Atletico": "ATM",
    "Real Sociedad": "RSO", "Real Betis": "BET", "Villarreal": "VIL",
    "Athletic Bilbao": "ATH", "Athletic Club": "ATH",
    "Sevilla": "SEV", "Valencia": "VAL", "Osasuna": "OSA",
    "Celta Vigo": "CEL", "Getafe": "GET", "Mallorca": "MLL",
    "Girona": "GIR", "Las Palmas": "LPA", "Alaves": "ALA",
    "Rayo Vallecano": "RAY", "Espanyol": "ESP", "Leganes": "LEG",
    "Real Valladolid": "VLL",
    # Serie A
    "AC Milan": "MIL", "Inter Milan": "INT", "Internazionale": "INT",
    "Juventus": "JUV", "Napoli": "NAP", "AS Roma": "ROM", "Roma": "ROM",
    "Lazio": "LAZ", "Atalanta": "ATA", "Fiorentina": "FIO",
    "Bologna": "BOL", "Torino": "TOR", "Udinese": "UDI",
    "Monza": "MON", "Empoli": "EMP", "Cagliari": "CAG",
    "Genoa": "GEN", "Lecce": "LEC", "Como": "COM",
    "Parma": "PAR", "Venezia": "VEN", "Hellas Verona": "VER",
    # Bundesliga
    "Bayern Munich": "BAY", "Bayern": "BAY", "FC Bayern Munich": "BAY",
    "Borussia Dortmund": "BVB", "Dortmund": "BVB",
    "RB Leipzig": "RBL", "Leipzig": "RBL",
    "Bayer Leverkusen": "LEV", "Leverkusen": "LEV",
    "Eintracht Frankfurt": "SGE", "Frankfurt": "SGE",
    "VfB Stuttgart": "STU", "Stuttgart": "STU",
    "Wolfsburg": "WOB", "VfL Wolfsburg": "WOB",
    "Freiburg": "FRE", "SC Freiburg": "FRE",
    "Union Berlin": "FCU", "Werder Bremen": "SVW", "Bremen": "SVW",
    "Hoffenheim": "TSG", "Mainz": "M05", "Mainz 05": "M05",
    "Augsburg": "FCA", "Heidenheim": "HDH",
    "Borussia Monchengladbach": "BMG", "Gladbach": "BMG",
    "St. Pauli": "STP", "Holstein Kiel": "KIE",
    "Bochum": "BOC", "VfL Bochum": "BOC",
    # Ligue 1
    "Paris Saint-Germain": "PSG", "PSG": "PSG",
    "Marseille": "MAR", "Olympique Marseille": "MAR",
    "Lyon": "LYO", "Olympique Lyon": "LYO",
    "Monaco": "MON", "AS Monaco": "MON",
    "Lille": "LIL", "LOSC Lille": "LIL",
    "Nice": "NIC", "OGC Nice": "NIC",
    "Lens": "LEN", "RC Lens": "LEN",
    "Rennes": "REN", "Stade Rennais": "REN",
    "Strasbourg": "STR", "RC Strasbourg": "STR",
    "Toulouse": "TLS", "Nantes": "NAN", "FC Nantes": "NAN",
    "Montpellier": "MTP", "Reims": "REI", "Stade Reims": "REI",
    "Brest": "BRS", "Stade Brestois": "BRS",
    "Le Havre": "HAV", "Angers": "ANG", "Auxerre": "AUX",
    "Saint-Etienne": "STE",
    # MLS
    "Inter Miami": "MIA", "LAFC": "LAF", "Los Angeles FC": "LAF",
    "LA Galaxy": "LAG", "Atlanta United": "ATL",
    "New York Red Bulls": "NYRB", "Red Bulls": "NYRB",
    "NYCFC": "NYC", "New York City FC": "NYC",
    "Seattle Sounders": "SEA", "Portland Timbers": "POR",
    "Columbus Crew": "CLB", "Cincinnati": "CIN", "FC Cincinnati": "CIN",
    "Nashville SC": "NSH", "Charlotte FC": "CLT",
    "Chicago Fire": "CHI", "CF Montreal": "MTL", "Montreal": "MTL",
    "Toronto FC": "TFC", "Philadelphia Union": "PHI",
    "D.C. United": "DCU", "DC United": "DCU",
    "New England Revolution": "NE", "Orlando City": "ORL",
    "Austin FC": "ATX", "FC Dallas": "DAL",
    "Houston Dynamo": "HOU", "Sporting KC": "SKC",
    "Minnesota United": "MIN", "Colorado Rapids": "COL",
    "Real Salt Lake": "RSL", "San Jose Earthquakes": "SJ",
    "Vancouver Whitecaps": "VAN", "St. Louis City": "STL",
}


def normalize_team_name(name: str, sport: str = "nba") -> str:
    """
    Normalize an external team name to its Kalshi ticker code.
    Returns the original name if no mapping is found.
    """
    team_map = NBA_TEAM_MAP if sport == "nba" else SOCCER_TEAM_MAP
    # Try exact match first
    if name in team_map:
        return team_map[name]
    # Try case-insensitive
    name_lower = name.lower()
    for key, val in team_map.items():
        if key.lower() == name_lower:
            return val
    # Return original if no match
    return name


def match_game_to_kalshi_tickers(
    home_code: str,
    away_code: str,
    kalshi_markets: List[Dict],
    has_draw: bool = False,
) -> List[SportsMarketOutcome]:
    """
    Match a game's team codes to Kalshi market tickers.

    Kalshi NBA tickers: KXNBAGAME-26FEB26MIAPHI-MIA, -PHI
    Kalshi soccer tickers: KXEPLGAME-26MAR01ARSCHE-ARS, -TIE, -CHE

    We look for tickers containing both team codes.
    """
    outcomes = []
    home_upper = home_code.upper()
    away_upper = away_code.upper()

    for market in kalshi_markets:
        ticker = market.get("ticker", "")
        ticker_upper = ticker.upper()

        # Check if this ticker contains both team codes
        if home_upper not in ticker_upper or away_upper not in ticker_upper:
            # Also try: ticker might only have one team code if it's the
            # specific team market (e.g., -MIA at the end)
            pass

        yes_price = market.get("yes_price") or market.get("yes_bid") or market.get("yes_ask")
        no_price = market.get("no_price") or market.get("no_bid") or market.get("no_ask")
        if yes_price is None and no_price is None:
            continue
        if yes_price is not None and no_price is None:
            no_price = 100 - yes_price
        elif no_price is not None and yes_price is None:
            yes_price = 100 - no_price

        yes_ask = market.get("yes_ask") or yes_price
        no_ask = market.get("no_ask") or no_price
        volume = market.get("volume", 0)

        # Determine outcome from ticker suffix
        # Tickers typically end with -TEAMCODE or -TIE
        parts = ticker.split("-")
        if len(parts) < 2:
            continue

        suffix = parts[-1].upper()

        # Check if this market belongs to our game
        # The middle part should contain both team codes
        game_part = "-".join(parts[:-1]).upper()
        if home_upper not in game_part and away_upper not in game_part:
            # Try checking if both codes appear anywhere in the full ticker
            if not (home_upper in ticker_upper and away_upper in ticker_upper):
                continue

        if suffix == home_upper:
            outcome = "home"
        elif suffix == away_upper:
            outcome = "away"
        elif suffix == "TIE":
            outcome = "draw"
        else:
            # Suffix doesn't match known outcomes for this game
            continue

        outcomes.append(SportsMarketOutcome(
            ticker=ticker,
            outcome=outcome,
            yes_price=yes_price,
            no_price=no_price,
            yes_ask=yes_ask,
            no_ask=no_ask,
            volume=volume,
        ))

    return outcomes


# ============================================================
# Odds conversion utilities
# ============================================================

def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def decimal_to_prob(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if odds <= 1.0:
        return 1.0
    return 1.0 / odds


def normalize_probs(probs: List[float]) -> List[float]:
    """Normalize probabilities to sum to 1.0."""
    total = sum(probs)
    if total <= 0:
        return probs
    return [p / total for p in probs]


# ============================================================
# Consensus algorithm (probability-spread)
# ============================================================

def compute_sports_consensus(
    forecast: SportsMultiSourceForecast,
    min_sources: int = 2,
) -> Optional[SportsConsensusResult]:
    """
    Compute consensus probabilities from multi-source sports forecasts.

    Algorithm (probability-spread, not clustering):
      1. Collect win probabilities from all sources
      2. If fewer than min_sources -> None
      3. Compute max spread across all outcomes
      4. Map spread to confidence level
      5. Consensus = median of each outcome's probabilities

    Returns None if insufficient sources or low confidence.
    """
    sources = forecast.sources
    n = len(sources)

    if n < min_sources:
        return None

    home_probs = [s.home_win for s in sources]
    away_probs = [s.away_win for s in sources]
    draw_probs = [s.draw for s in sources]

    max_spread = max(
        max(home_probs) - min(home_probs),
        max(away_probs) - min(away_probs),
        (max(draw_probs) - min(draw_probs)) if any(d > 0 for d in draw_probs) else 0,
    )

    # Confidence thresholds (wider for 2-source consensus)
    if max_spread < 0.12:
        confidence = "high"
    elif max_spread < 0.22:
        confidence = "medium"
    else:
        confidence = "low"

    # Consensus = median of each outcome
    home_consensus = median(home_probs)
    away_consensus = median(away_probs)
    draw_consensus = median(draw_probs) if any(d > 0 for d in draw_probs) else 0.0

    # Normalize to sum to 1
    total = home_consensus + away_consensus + draw_consensus
    if total > 0:
        home_consensus /= total
        away_consensus /= total
        draw_consensus /= total

    return SportsConsensusResult(
        home_prob=home_consensus,
        away_prob=away_consensus,
        draw_prob=draw_consensus,
        confidence=confidence,
        max_spread=max_spread,
        source_count=n,
    )


# ============================================================
# Signal generation (direct probability comparison + Kelly)
# ============================================================

def generate_sports_signals(
    outcomes: List[SportsMarketOutcome],
    consensus: SportsConsensusResult,
    game_desc: str,
    bankroll: float,
    min_edge: float = 0.08,
    max_position_pct: float = 0.05,
    kelly_fraction: float = 0.5,
    rationale_prefix: str = "SPORTS",
) -> List[SportsTradeSignal]:
    """
    Generate trade signals by comparing consensus probabilities to market prices.

    For each outcome market (home/away/draw), checks if our consensus probability
    exceeds the market price by min_edge. Uses Kelly criterion for sizing.
    """
    signals = []

    prob_map = {
        "home": consensus.home_prob,
        "away": consensus.away_prob,
        "draw": consensus.draw_prob,
    }

    for mkt in outcomes:
        our_prob = prob_map.get(mkt.outcome, 0.0)
        market_prob = mkt.implied_prob

        if our_prob <= 0.01 or market_prob <= 0.01 or market_prob >= 0.99:
            continue

        # Calculate edge in both directions
        yes_edge = our_prob - market_prob
        no_edge = market_prob - our_prob

        if yes_edge >= min_edge:
            side = "YES"
            edge = yes_edge
            odds = (1 - market_prob) / market_prob if market_prob > 0 else 0
            win_prob = our_prob
        elif no_edge >= min_edge:
            side = "NO"
            edge = no_edge
            odds = market_prob / (1 - market_prob) if market_prob < 1 else 0
            win_prob = 1 - our_prob
        else:
            continue

        # Kelly criterion
        if odds > 0:
            kelly = (odds * win_prob - (1 - win_prob)) / odds
            kelly = max(0, kelly) * kelly_fraction
        else:
            kelly = 0

        position_size = min(
            kelly * bankroll,
            max_position_pct * bankroll,
        )

        # Fee-adjusted edge
        if side == "YES":
            entry_price_cents = mkt.yes_ask if mkt.yes_ask > 0 else mkt.yes_price
        else:
            entry_price_cents = mkt.no_ask if mkt.no_ask > 0 else mkt.no_price

        fee = kalshi_taker_fee(entry_price_cents)
        net_edge = edge - fee

        if net_edge < 0.03:
            continue

        entry_price_dollars = entry_price_cents / 100.0
        if entry_price_dollars <= 0:
            continue

        shares = max(1, int(position_size / entry_price_dollars))
        shares = min(shares, 10)
        actual_position = shares * entry_price_dollars

        # Limit price
        if side == "YES":
            limit_price = min(entry_price_cents, int(our_prob * 100) - 1)
            limit_price = max(1, limit_price)
        else:
            limit_price = min(entry_price_cents, int((1 - our_prob) * 100) - 1)
            limit_price = max(1, limit_price)

        signal = SportsTradeSignal(
            market=mkt,
            our_prob=our_prob,
            market_prob=market_prob,
            edge=edge,
            side=side,
            confidence=min(1.0, our_prob + 0.1),
            limit_price=limit_price,
            position_size_dollars=actual_position,
            shares=shares,
            game_desc=game_desc,
            rationale=(
                f"{rationale_prefix}: {game_desc} {mkt.outcome} — "
                f"prob {our_prob:.0%} vs market {market_prob:.0%} = "
                f"{edge:.0%} edge ({side}), net: {net_edge:.0%}"
            ),
        )
        signals.append(signal)

    signals.sort(key=lambda s: s.edge, reverse=True)
    return signals


# ============================================================
# Trade execution
# ============================================================

async def execute_sports_trade(
    signal: SportsTradeSignal,
    kalshi_client: KalshiClient,
    db_manager: DatabaseManager,
    strategy: str = "sports",
) -> bool:
    """
    Execute a single sports trade signal using a limit order.
    Returns True if order was placed successfully.
    """
    try:
        existing = await db_manager.get_position_by_market_and_side(
            signal.market.ticker, signal.side,
        )
        if existing:
            logger.info(
                f"SKIP: Already hold {signal.market.ticker} {signal.side}"
            )
            return False

        client_order_id = str(uuid.uuid4())
        side_lower = signal.side.lower()

        order_kwargs = {
            "ticker": signal.market.ticker,
            "client_order_id": client_order_id,
            "side": side_lower,
            "action": "buy",
            "count": signal.shares,
            "type_": "limit",
        }

        if side_lower == "yes":
            order_kwargs["yes_price"] = signal.limit_price
        else:
            order_kwargs["no_price"] = signal.limit_price

        logger.info(
            f"SPORTS TRADE: {signal.game_desc} {signal.market.ticker} — "
            f"{signal.shares} {signal.side} @ {signal.limit_price}c "
            f"(edge: {signal.edge:.0%})"
        )

        order_response = await kalshi_client.place_order(**order_kwargs)

        if order_response and "order" in order_response:
            order_id = order_response["order"].get("order_id", client_order_id)

            position = Position(
                market_id=signal.market.ticker,
                side=signal.side,
                quantity=signal.shares,
                entry_price=signal.limit_price / 100.0,
                live=True,
                timestamp=datetime.now(),
                rationale=signal.rationale,
                strategy=strategy,
                stop_loss_price=0.01,
                take_profit_price=0.99,
                max_hold_hours=48,
            )
            await db_manager.add_position(position)

            logger.info(
                f"SPORTS ORDER PLACED: {signal.market.ticker} — "
                f"Order ID: {order_id}, {signal.shares} {signal.side} @ {signal.limit_price}c"
            )
            return True
        else:
            logger.error(f"Sports order failed: {order_response}")
            return False

    except Exception as e:
        logger.error(f"Error executing sports trade: {e}")
        return False
