import numpy as np
import sounddevice as sd
import json
import asyncio
import websockets
import random
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Card:
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['H', 'D', 'C', 'S']

    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f"{self.rank}{self.suit}"

class Deck:
    def __init__(self):
        self.cards = [Card(rank, suit) for suit in Card.suits for rank in Card.ranks]
        random.shuffle(self.cards)

    def deal(self):
        return self.cards.pop()

class PokerHand:
    def __init__(self, cards):
        self.cards = cards

    def evaluate(self):
        ranks = [card.rank for card in self.cards]
        suits = [card.suit for card in self.cards]
        unique_ranks = set(ranks)
        
        if len(unique_ranks) == 1:
            return "Five of a Kind", 9
        elif len(set(suits)) == 1:
            return "Flush", 6
        else:
            return "High Card", 1

# Reusing existing PokerGameSimulator with updated methods
class PokerGameSimulator:
    def __init__(self, num_players: int = 4):
        self.num_players = num_players
        self.deck = Deck()
        self.players = self._initialize_players()
        self.pot = 0
        self.community_cards = []
        self.game_phase = "preflop"

    def _initialize_players(self) -> List[Dict]:
        return [
            {
                "id": i,
                "name": f"Player_{i}",
                "chips": 1000,
                "hand": [self.deck.deal(), self.deck.deal()],
                "bet": 0,
                "active": True
            }
            for i in range(self.num_players)
        ]

    def simulate_action(self) -> Dict:
        action = random.choice(["bet", "call", "raise", "fold"])
        amount = random.randint(10, 100) if action in ["bet", "raise"] else 0
        return {
            "action": action,
            "amount": amount,
            "phase": self.game_phase,
            "pot": self.pot + amount,
            "timestamp": datetime.now().isoformat()
        }

# Using enhanced AudioEncoder from existing codebase
class AudioEncoder:
    def __init__(self, sample_rate: int = 44100, duration: float = 1.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.frequencies = {
            "bet": 440,   # A4
            "call": 554,  # C#5
            "raise": 659, # E5
            "fold": 349   # F4
        }
        self.error_correction_code = {
            "bet": [1, 0, 1],
            "call": [1, 1, 0],
            "raise": [0, 1, 1],
            "fold": [1, 0, 0]
        }

    def generate_tone(self, freq: float, duration: Optional[float] = None) -> np.ndarray:
        t = np.linspace(0, duration or self.duration, 
                       int(self.sample_rate * (duration or self.duration)), False)
        return 0.5 * np.sin(2 * np.pi * freq * t)

    def encode_game_state(self, game_state: Dict) -> np.ndarray:
        action = game_state["action"]
        base_freq = self.frequencies[action]
        amount = game_state["amount"]
        modulated_freq = base_freq * (1 + amount / 1000.0)
        tone = self.generate_tone(modulated_freq)
        error_code = self.error_correction_code[action]
        correction_signal = self.generate_correction_tones(error_code)
        return np.concatenate([tone, correction_signal])

    def generate_correction_tones(self, error_code: List[int]) -> np.ndarray:
        correction_tones = []
        for bit in error_code:
            freq = 2000 if bit == 1 else 1000
            correction_tones.append(self.generate_tone(freq, duration=0.1))
        return np.concatenate(correction_tones)

    def decode_audio(self, audio_signal: np.ndarray) -> Dict:
        fft = np.fft.fft(audio_signal)
        freqs = np.fft.fftfreq(len(audio_signal), 1/self.sample_rate)
        magnitude = np.abs(fft)
        dom_freq = freqs[np.argmax(magnitude)]
        
        action = min(self.frequencies.items(), 
                    key=lambda x: abs(x[1] - abs(dom_freq)))[0]
        
        base_freq = self.frequencies[action]
        amount = int(((abs(dom_freq) / base_freq) - 1) * 1000)
        
        return {
            "action": action,
            "amount": max(0, amount),
            "decoded_frequency": abs(dom_freq)
        }

async def game_state_handler(websocket, path):
    simulator = PokerGameSimulator()
    encoder = AudioEncoder()
    metrics = MetricsAnalyzer()
    
    while True:
        game_state = simulator.simulate_action()
        encoded_signal = encoder.encode_game_state(game_state)
        
        # Simulate audio playback and capture
        sd.play(encoded_signal, encoder.sample_rate)
        sd.wait()
        
        captured_signal = sd.rec(
            int(encoder.sample_rate * encoder.duration * 1.5),
            samplerate=encoder.sample_rate,
            channels=1
        )
        sd.wait()
        
        decoded_state = encoder.decode_audio(captured_signal)
        comparison = metrics.compare_states(game_state, decoded_state)
        await websocket.send(json.dumps(comparison))
        await asyncio.sleep(1)

async def main():
    server = await websockets.serve(game_state_handler, "localhost", 8765)
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())