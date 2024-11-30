import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import soundfile as sf
import json

@dataclass
class GameState:
    game_id: str
    players: List[Dict]
    pot: int
    current_round: str
    winner: Optional[Dict] = None

class GameAudioEncoder:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.base_frequencies = {
            'game_start': 440.0,  # A4
            'betting': 554.37,    # C#5
            'showdown': 659.25,   # E5
            'victory': 880.0      # A5
        }
        self.data_dir = Path("game_data")
        self.data_dir.mkdir(exist_ok=True)
        self.audio_buffer = np.array([])
        self.caption_buffer = []

    def encode_game_state(self, state: GameState) -> np.ndarray:
        """Convert game state to musical audio"""
        duration = 0.5  # seconds per state
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Base frequency for game identification
        game_freq = self.base_frequencies['game_start'] * (1 + int(state.game_id) / 1000)
        
        # Encode pot size into amplitude
        amplitude = min(0.8, 0.2 + (state.pot / 10000))
        
        # Generate harmonics based on player states
        harmonics = []
        for player in state.players:
            player_freq = game_freq * (1 + player['chips'] / 10000)
            harmonic = 0.3 * np.sin(2 * np.pi * player_freq * t)
            harmonics.append(harmonic)
        
        # Combine signals
        base_signal = amplitude * np.sin(2 * np.pi * game_freq * t)
        combined_signal = base_signal + sum(harmonics)
        
        # Normalize
        combined_signal = combined_signal / np.max(np.abs(combined_signal))
        
        # Apply envelope
        envelope = np.exp(-3 * t/duration)
        return combined_signal * envelope

    def create_caption(self, state: GameState) -> str:
        """Generate human-readable caption for game state"""
        caption = f"Game {state.game_id} - {state.current_round}\n"
        caption += f"Pot: {state.pot} chips\n"
        
        for player in state.players:
            caption += f"{player['name']}: {player['chips']} chips\n"
        
        if state.winner:
            caption += f"Winner: {state.winner['name']} (+{state.pot} chips)\n"
        
        return caption

    def process_game_state(self, state: GameState):
        """Process game state into audio and captions"""
        # Generate audio
        audio_segment = self.encode_game_state(state)
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_segment])
        
        # Generate caption
        caption = self.create_caption(state)
        self.caption_buffer.append(caption)
        
        # Save state if buffer gets too large
        if len(self.audio_buffer) > self.sample_rate * 60:  # Save every minute
            self.save_current_buffer()

    def save_current_buffer(self):
        """Save accumulated audio and captions"""
        if len(self.audio_buffer) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save audio
            audio_path = self.data_dir / f"game_audio_{timestamp}.wav"
            sf.write(str(audio_path), self.audio_buffer, self.sample_rate)
            
            # Save captions
            caption_path = self.data_dir / f"game_captions_{timestamp}.txt"
            with caption_path.open('w') as f:
                f.write('\n'.join(self.caption_buffer))
            
            # Reset buffers
            self.audio_buffer = np.array([])
            self.caption_buffer = []

class EnhancedGameManager:
    def __init__(self, batch_size: int = 100):
        self.audio_encoder = GameAudioEncoder()
        # ... rest of initialization ...

    async def _run_table(self, table, game_id: str) -> None:
        try:
            # Create initial game state
            state = GameState(
                game_id=game_id,
                players=[{
                    'name': p.name,
                    'chips': p.chips,
                    'persona': p.historical_persona
                } for p in table.players],
                pot=0,
                current_round="Pre-flop"
            )
            
            # Record initial state
            self.audio_encoder.process_game_state(state)
            
            # Run game rounds
            rounds = [("Flop", 3), ("Turn", 1), ("River", 1)]
            for round_name, _ in rounds:
                # Update state after each round
                state.current_round = round_name
                state.pot = table.pot
                state.players = [{
                    'name': p.name,
                    'chips': p.chips
                } for p in table.players]
                self.audio_encoder.process_game_state(state)
            
            # Record final state with winner
            winner, _, _ = table.determine_winner()
            if winner:
                state.winner = {
                    'name': winner.name,
                    'chips': winner.chips
                }
                self.audio_encoder.process_game_state(state)
            
        except Exception as e:
            logger.error(f"Error running table {game_id}: {e}")

        finally:
            # Save any remaining data
            self.audio_encoder.save_current_buffer()