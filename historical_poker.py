from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4
import random
import numpy as np
import soundfile as sf

# Add Suit enum at the top
class Suit(str, Enum):
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"

# First, define the Card class
class Card(BaseModel):
    suit: Suit
    value: int = Field(ge=2, le=14)
    historical_significance: Optional[str] = None
    
    def __str__(self):
        values = {11: "J", 12: "Q", 13: "K", 14: "A"}
        return f"{values.get(self.value, self.value)}{self.suit}"

# Then define the Player class
class Player(BaseModel):
    name: str
    chips: int = 1000
    cards: List[Card] = Field(default_factory=list)
    historical_persona: Optional[str] = None
    fate_points: int = Field(default=3)

    def make_bet(self, amount: int) -> bool:
        if amount <= self.chips:
            self.chips -= amount
            return True
        return False

# Define OmniscientComment before PokerTable
class OmniscientComment(BaseModel):
    prophecy: str
    historical_parallel: str
    architectural_significance: str

# Define HistoricalContext before PokerTable
class HistoricalContext(BaseModel):
    era: str
    civilization: str
    significant_events: List[str]
    cultural_context: str
    architectural_setting: str

class GamePhase(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"

class PlayerAction(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    ALL_IN = "all_in"
    USE_FATE = "use_fate"

class HandRank(str, Enum):
    HIGH_CARD = "high_card"
    PAIR = "pair"
    TWO_PAIR = "two_pair"
    THREE_KIND = "three_of_a_kind"
    STRAIGHT = "straight"
    FLUSH = "flush"
    FULL_HOUSE = "full_house"
    FOUR_KIND = "four_of_a_kind"
    STRAIGHT_FLUSH = "straight_flush"
    ROYAL_FLUSH = "royal_flush"

class PokerTable(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    players: List[Player]
    deck: List[Card] = Field(default_factory=list)
    pot: int = 0
    current_bet: int = 0
    community_cards: List[Card] = Field(default_factory=list)
    historical_context: HistoricalContext
    oracle_commentary: List[OmniscientComment] = Field(default_factory=list)
    current_phase: GamePhase = GamePhase.PREFLOP
    active_player_index: int = 0
    
    def deal_flop(self):
        if len(self.community_cards) == 0:
            self.community_cards = [self.deck.pop() for _ in range(3)]
            self.current_phase = GamePhase.FLOP
            self._add_historical_commentary("The flop reveals the first glimpse of destiny")
    
    def deal_turn(self):
        if len(self.community_cards) == 3:
            self.community_cards.append(self.deck.pop())
            self.current_phase = GamePhase.TURN
            self._add_historical_commentary("The turn card shifts the tides of fate")
    
    def deal_river(self):
        if len(self.community_cards) == 4:
            self.community_cards.append(self.deck.pop())
            self.current_phase = GamePhase.RIVER
            self._add_historical_commentary("The river completes the sacred geometric pattern")
    
    def evaluate_hand(self, player: Player) -> tuple[HandRank, List[Card]]:
        all_cards = player.cards + self.community_cards
        all_cards.sort(key=lambda x: x.value, reverse=True)
        
        # Check for flush
        suits_count = {}
        for card in all_cards:
            suits_count[card.suit] = suits_count.get(card.suit, 0) + 1
        
        flush_suit = next((suit for suit, count in suits_count.items() if count >= 5), None)
        flush_cards = [card for card in all_cards if card.suit == flush_suit] if flush_suit else []
        
        # Check for straight
        values = sorted(set(card.value for card in all_cards))
        straight_high = None
        for i in range(len(values) - 4):
            if values[i] - values[i + 4] == 4:
                straight_high = values[i]
                break
        
        # Special case: Ace-low straight
        if not straight_high and 14 in values and all(v in values for v in [2, 3, 4, 5]):
            straight_high = 5
        
        if flush_suit and straight_high:
            flush_values = sorted(set(card.value for card in flush_cards))
            for i in range(len(flush_values) - 4):
                if flush_values[i] - flush_values[i + 4] == 4:
                    if flush_values[i] == 14:
                        return HandRank.ROYAL_FLUSH, flush_cards[:5]
                    return HandRank.STRAIGHT_FLUSH, flush_cards[:5]
        
        # Count values
        value_count = {}
        for card in all_cards:
            value_count[card.value] = value_count.get(card.value, 0) + 1
        
        # Four of a kind
        four_kind = next((v for v, c in value_count.items() if c == 4), None)
        if four_kind:
            cards = [c for c in all_cards if c.value == four_kind]
            kicker = next(c for c in all_cards if c.value != four_kind)
            return HandRank.FOUR_KIND, cards + [kicker]
        
        # Full house
        three_kind = next((v for v, c in value_count.items() if c == 3), None)
        if three_kind:
            pair = next((v for v, c in value_count.items() if c >= 2 and v != three_kind), None)
            if pair:
                cards = [c for c in all_cards if c.value == three_kind or c.value == pair][:5]
                return HandRank.FULL_HOUSE, cards
        
        # Flush
        if flush_cards:
            return HandRank.FLUSH, flush_cards[:5]
        
        # Straight
        if straight_high:
            cards = []
            if straight_high == 5 and 14 in values:  # Ace-low straight
                cards = [c for c in all_cards if c.value in [14, 2, 3, 4, 5]]
            else:
                target_values = range(straight_high, straight_high - 5, -1)
                cards = [next(c for c in all_cards if c.value == v) for v in target_values]
            return HandRank.STRAIGHT, cards
        
        # Three of a kind
        if three_kind:
            cards = [c for c in all_cards if c.value == three_kind]
            kickers = [c for c in all_cards if c.value != three_kind][:2]
            return HandRank.THREE_KIND, cards + kickers
        
        # Two pair
        pairs = [v for v, c in value_count.items() if c == 2]
        if len(pairs) >= 2:
            pairs = sorted(pairs, reverse=True)[:2]
            cards = [c for c in all_cards if c.value in pairs]
            kicker = next(c for c in all_cards if c.value not in pairs)
            return HandRank.TWO_PAIR, cards + [kicker]
        
        # Pair
        pair = next((v for v, c in value_count.items() if c == 2), None)
        if pair:
            cards = [c for c in all_cards if c.value == pair]
            kickers = [c for c in all_cards if c.value != pair][:3]
            return HandRank.PAIR, cards + kickers
        
        # High card
        return HandRank.HIGH_CARD, all_cards[:5]

    def process_action(self, player: Player, action: PlayerAction, amount: Optional[int] = None):
        if action == PlayerAction.FOLD:
            player.cards = []
            self._add_historical_commentary(f"{player.historical_persona} retreats from the battlefield")
        
        elif action == PlayerAction.CALL:
            call_amount = self.current_bet
            player.make_bet(call_amount)
            self.pot += call_amount
        
        elif action == PlayerAction.RAISE:
            if amount and amount > self.current_bet:
                player.make_bet(amount)
                self.pot += amount
                self.current_bet = amount
                self._add_historical_commentary(f"{player.historical_persona} makes a bold move")
        
        elif action == PlayerAction.USE_FATE:
            if card := self.use_fate_point(player):
                player.cards.append(card)
                self._add_historical_commentary(f"{player.historical_persona} alters their destiny")
    
    def _add_historical_commentary(self, event: str):
        comment = OmniscientComment(
            prophecy=f"The fates decree: {event}",
            historical_parallel=random.choice(self.historical_context.significant_events),
            architectural_significance=f"The {self.historical_context.architectural_setting} bears witness"
        )
        self.oracle_commentary.append(comment)

    def determine_winner(self) -> tuple[Player, HandRank, List[Card]]:
        active_players = [p for p in self.players if p.cards]
        if not active_players:
            return None, None, []
        
        player_hands = [(p, *self.evaluate_hand(p)) for p in active_players]
        return max(player_hands, key=lambda x: (x[1].value, [c.value for c in x[2]]))

    def initialize_deck(self):
        self.deck = [
            Card(
                suit=suit,
                value=value,
                historical_significance=self._get_card_significance(suit, value)
            )
            for suit in Suit
            for value in range(2, 15)
        ]
        random.shuffle(self.deck)
    
    def _get_card_significance(self, suit: Suit, value: int) -> str:
        significances = {
            (Suit.SPADES, 14): "The Sword of Damocles",
            (Suit.HEARTS, 13): "Cleopatra's Heart",
            # Add more significances as needed
        }
        return significances.get((suit, value), "A witness to history")

    def deal_cards(self):
        """Add this new method that was referenced but missing"""
        for player in self.players:
            player.cards = [self.deck.pop() for _ in range(2)]

    def use_fate_point(self, player: Player) -> Optional[Card]:
        """Use a fate point to draw an extra card if available"""
        if player.fate_points > 0:
            player.fate_points -= 1
            if self.deck:
                return self.deck.pop()
        return None

class AudioInterface:
    SAMPLE_RATE = 44100
    
    # Frequency mappings for different game events
    FREQUENCIES = {
        'deal': 440,    # A4
        'bet': 550,     # C#5
        'call': 660,    # E5
        'raise': 880,   # A5
        'fold': 330,    # E4
        'win': 1100,    # C#6
        'fate': 1320    # E6
    }
    
    def __init__(self):
        self.buffer = []
    
    def add_to_buffer(self, sound_data):
        """Add sound data to buffer"""
        if sound_data is not None:
            self.buffer.append(sound_data)
    
    def card_to_sound(self, card: Card):
        """Convert card to sound data"""
        # Simple implementation - just return None for now
        return None
    
    def action_to_sound(self, action: PlayerAction, amount: Optional[int] = None):
        """Convert player action to sound data"""
        # Simple implementation - just return None for now
        return None
    
    def play_hand_result(self, rank: HandRank):
        """Generate sound for hand result"""
        # Simple implementation - just return None for now
        return None

class AudioPokerTable(PokerTable):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow non-Pydantic types
    audio: AudioInterface = None  # Add this field definition
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio = AudioInterface()
    
    def process_action(self, player: Player, action: PlayerAction, amount: Optional[int] = None):
        super().process_action(player, action, amount)
        self.audio.add_to_buffer(self.audio.action_to_sound(action, amount))
    
    def deal_cards(self):
        for card in self.deck[-2:]:  # Deal last two cards
            self.audio.add_to_buffer(self.audio.card_to_sound(card))
        super().deal_cards()
    
    def determine_winner(self):
        winner, rank, cards = super().determine_winner()
        if winner:
            self.audio.add_to_buffer(self.audio.play_hand_result(rank))
        return winner, rank, cards

def create_historical_game(players_data: List[Dict]) -> AudioPokerTable:
    players = [
        Player(
            name=p["name"],
            chips=p.get("chips", 1000),
            historical_persona=p["persona"]
        )
        for p in players_data
    ]
    
    context = HistoricalContext(
        era="Classical Antiquity",
        civilization="Multi-Epochal",
        significant_events=[
            "The Fall of Constantinople",
            "The Building of the Pyramids",
            "The Signing of the Magna Carta"
        ],
        cultural_context="A convergence of great civilizations",
        architectural_setting="The Great Library of Alexandria"
    )
    
    table = AudioPokerTable(players=players, historical_context=context)
    table.initialize_deck()
    table.deal_cards()
    
    return table

class PlayerInterface:
    def __init__(self, player: Player):
        self.player = player
        self._action_history: List[tuple[PlayerAction, int]] = []

    def get_info(self) -> dict:
        """Get current player information"""
        return {
            "name": self.player.name,
            "chips": self.player.chips,
            "cards": self.player.cards,
            "persona": self.player.historical_persona,
            "fate_points": self.player.fate_points
        }

    def make_action(self, action: PlayerAction, amount: Optional[int] = None) -> tuple[PlayerAction, Optional[int]]:
        """Execute a player action and record it"""
        if not self._can_perform_action(action, amount):
            raise ValueError(f"Invalid action: {action} with amount {amount}")
        
        self._action_history.append((action, amount if amount else 0))
        return action, amount

    def _can_perform_action(self, action: PlayerAction, amount: Optional[int] = None) -> bool:
        """Validate if player can perform the action"""
        if action == PlayerAction.USE_FATE and self.player.fate_points <= 0:
            return False
        if action in [PlayerAction.RAISE, PlayerAction.CALL] and amount and amount > self.player.chips:
            return False
        return True

    def get_action_history(self) -> List[tuple[PlayerAction, int]]:
        """Get player's action history"""
        return self._action_history.copy()

    def add_card(self, card: Card):
        """Add a card to player's hand"""
        self.player.cards.append(card)

    def clear_hand(self):
        """Clear player's current hand"""
        self.player.cards = []

    def add_chips(self, amount: int):
        """Add chips to player's stack"""
        self.player.chips += amount

    def use_fate_point(self) -> bool:
        """Use a fate point if available"""
        if self.player.fate_points > 0:
            self.player.fate_points -= 1
            return True
        return False

# Example usage:
if __name__ == "__main__":
    player = Player(
        name="Julius Caesar",
        historical_persona="Roman Emperor",
        chips=2000
    )
    
    interface = PlayerInterface(player)
    
    # Get player info
    print(interface.get_info())
    
    # Make some actions
    try:
        interface.make_action(PlayerAction.RAISE, 100)
        interface.make_action(PlayerAction.CALL, 50)
        
        # Try to use fate point
        if interface.use_fate_point():
            print("Fate point used successfully")
        
        # View action history
        print("Action history:", interface.get_action_history())
        
    except ValueError as e:
        print(f"Error: {e}")