import numpy as np
import soundfile as sf
from typing import List, Optional
from historical_poker import PlayerAction, Card, HandRank
import pickle
from pathlib import Path
from scipy import signal
from dataclasses import dataclass
from typing import Dict, Tuple, Any
import time

@dataclass
class GameAudioPattern:
    audio_data: np.ndarray
    metadata: Dict[str, Any]
    timestamp: float
    pattern_type: str

class AudioInterface:
    SAMPLE_RATE = 44100
    DURATION = 0.3  # Duration of each sound in seconds
    
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
        self.streams = {}
        self.buffer = []
        self.pattern_database: Dict[str, GameAudioPattern] = {}
        self.recording_path = Path("audio_patterns")
        self.recording_path.mkdir(exist_ok=True)
        self._load_patterns()

    def _load_patterns(self):
        """Load existing patterns from disk"""
        pattern_file = self.recording_path / "patterns.pkl"
        if pattern_file.exists():
            with open(pattern_file, "rb") as f:
                self.pattern_database = pickle.load(f)

    def save_pattern(self, audio_data: np.ndarray, metadata: Dict[str, Any], pattern_type: str):
        """Save a new audio pattern with metadata"""
        pattern = GameAudioPattern(
            audio_data=audio_data,
            metadata=metadata,
            timestamp=time.time(),
            pattern_type=pattern_type
        )
        
        # Generate unique key based on pattern type and timestamp
        key = f"{pattern_type}_{len(self.pattern_database)}"
        self.pattern_database[key] = pattern
        
        # Save to disk
        with open(self.recording_path / "patterns.pkl", "wb") as f:
            pickle.dump(self.pattern_database, f)

    def analyze_pattern(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Analyze audio pattern characteristics"""
        # Extract features like frequency components, amplitude envelope, etc.
        freqs, times, spectrogram = signal.spectrogram(audio_data, fs=self.SAMPLE_RATE)
        
        return {
            "mean_frequency": np.mean(freqs),
            "max_amplitude": np.max(np.abs(audio_data)),
            "duration": len(audio_data) / self.SAMPLE_RATE,
            "spectral_centroid": np.sum(freqs * np.mean(spectrogram, axis=1)) / np.sum(np.mean(spectrogram, axis=1))
        }

    def generate_tone(self, frequency: float, duration: float = DURATION) -> np.ndarray:
        """Generate a sine wave tone at the given frequency"""
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration))
        tone = np.sin(2 * np.pi * frequency * t)
        # Apply fade in/out to avoid clicks
        fade = 0.1
        fade_len = int(fade * self.SAMPLE_RATE)
        tone[:fade_len] *= np.linspace(0, 1, fade_len)
        tone[-fade_len:] *= np.linspace(1, 0, fade_len)
        return tone

    def card_to_sound(self, card: Card) -> np.ndarray:
        """Enhanced card_to_sound with pattern storage"""
        audio_data = super().card_to_sound(card)
        
        # Store the pattern
        metadata = {
            "card_value": card.value,
            "card_suit": card.suit,
            "historical_significance": card.historical_significance,
            "features": self.analyze_pattern(audio_data)
        }
        self.save_pattern(audio_data, metadata, "card_sound")
        
        return audio_data

    def action_to_sound(self, action_type: str, amount: Optional[int] = None) -> np.ndarray:
        """Generate sound for player actions"""
        # Map action types to frequencies from our FREQUENCIES dictionary
        if action_type not in self.FREQUENCIES:
            # Default to 'deal' frequency if action type not found
            base_freq = self.FREQUENCIES['deal']
        else:
            base_freq = self.FREQUENCIES[action_type]
        
        # Generate the basic action sound
        sound = self.generate_tone(base_freq)
        
        # If there's a bet amount, modify the sound
        if amount is not None and amount > 0:
            # Add a secondary tone based on bet size
            bet_freq = base_freq * (1 + (amount / 5000))  # Increase frequency with bet size
            bet_sound = self.generate_tone(bet_freq, duration=self.DURATION * 0.5)
            sound = np.concatenate([sound, bet_sound])
        
        return sound

    def play_hand_result(self, rank: HandRank) -> np.ndarray:
        """Generate victory sound based on hand rank"""
        base_freq = self.FREQUENCIES['win']
        rank_multiplier = {
            HandRank.HIGH_CARD: 1.0,
            HandRank.PAIR: 1.1,
            HandRank.TWO_PAIR: 1.2,
            HandRank.THREE_KIND: 1.3,
            HandRank.STRAIGHT: 1.4,
            HandRank.FLUSH: 1.5,
            HandRank.FULL_HOUSE: 1.6,
            HandRank.FOUR_KIND: 1.7,
            HandRank.STRAIGHT_FLUSH: 1.8,
            HandRank.ROYAL_FLUSH: 2.0
        }
        
        freq = base_freq * rank_multiplier[rank]
        fanfare = np.concatenate([
            self.generate_tone(freq),
            self.generate_tone(freq * 1.25),
            self.generate_tone(freq * 1.5)
        ])
        return fanfare

    def add_to_buffer(self, sound: np.ndarray):
        """Add a sound to the playback buffer"""
        if len(self.buffer) == 0:
            self.buffer = sound
        else:
            self.buffer = np.concatenate([self.buffer, sound])

    def save_buffer(self, filename: str = "game_audio.wav"):
        """Save the current audio buffer to a WAV file"""
        if len(self.buffer) > 0:
            sf.write(filename, self.buffer, self.SAMPLE_RATE)
            self.buffer = np.array([])  # Clear buffer after saving

    def clear_buffer(self):
        """Clear the audio buffer"""
        self.buffer = np.array([])

    def find_similar_patterns(self, audio_data: np.ndarray, pattern_type: str = None, threshold: float = 0.8) -> List[GameAudioPattern]:
        """Find similar patterns in the database"""
        target_features = self.analyze_pattern(audio_data)
        similar_patterns = []
        
        for key, pattern in self.pattern_database.items():
            if pattern_type and pattern.pattern_type != pattern_type:
                continue
                
            # Compare features
            pattern_features = pattern.metadata["features"]
            similarity = self._calculate_similarity(target_features, pattern_features)
            
            if similarity >= threshold:
                similar_patterns.append(pattern)
        
        return similar_patterns

    def _calculate_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Calculate similarity between two feature sets"""
        differences = []
        for key in features1:
            if key in features2:
                diff = abs(features1[key] - features2[key]) / max(features1[key], features2[key])
                differences.append(1 - diff)
        
        return np.mean(differences) if differences else 0.0

    def initialize_streams(self):
        """Initialize different audio channels"""
        self.streams = {
            'cards': [],
            'chips': [],
            'actions': [],
            'markers': [],
            'commentary': []
        }
    
    def add_card_sound(self, card):
        """Generate and add card sound based on suit and value"""
        # Convert card attributes to frequency/amplitude
        frequency = self.card_to_frequency(card)
        amplitude = self.card_to_amplitude(card)
        self.streams['cards'].append((frequency, amplitude))
    
    def add_chip_sound(self, amount):
        """Generate chip sounds based on bet amount"""
        # Convert bet amount to sound parameters
        self.streams['chips'].append(self.amount_to_sound(amount))
    
    def add_action_sound(self, action_type: str, amount: Optional[int] = None):
        """Generate sound for player actions"""
        sound = self.action_to_sound(action_type, amount)
        self.streams['actions'].append(sound)
    
    def add_player_marker(self, player_name):
        """Add player turn indicator"""
        self.streams['markers'].append(self.name_to_marker(player_name))
    
    def add_round_marker(self, round_name):
        """Add round transition marker"""
        self.streams['markers'].append(self.round_to_marker(round_name))
    
    def save_game_audio(self, filename: str):
        """Mix all streams and save to file"""
        mixed_audio = self.mix_streams()
        sf.write(filename, mixed_audio, self.SAMPLE_RATE)

    def card_to_frequency(self, card) -> float:
        """Convert card to base frequency"""
        # Map card values (2-14) to musical scale
        base_freq = 220  # A3 note
        value_multiplier = 2 ** ((card.value - 2) / 12)  # Semitone steps
        
        # Suit affects the octave
        suit_multiplier = {
            'HEARTS': 1.0,    # Same octave
            'DIAMONDS': 1.25, # Major third up
            'CLUBS': 1.5,     # Perfect fifth up
            'SPADES': 2.0     # Octave up
        }
        
        return base_freq * value_multiplier * suit_multiplier[card.suit.name]

    def card_to_amplitude(self, card) -> float:
        """Convert card to sound amplitude"""
        # Higher cards are slightly louder
        base_amplitude = 0.5
        value_boost = (card.value - 2) / 24  # Small increase for higher cards
        
        # Face cards get extra amplitude
        if card.value > 10:
            value_boost += 0.1
        
        return min(base_amplitude + value_boost, 1.0)

    def amount_to_sound(self, amount: int) -> Tuple[float, float]:
        """Convert bet amount to sound parameters"""
        # Larger bets get higher frequencies and amplitudes
        base_freq = 330  # E4 note
        freq_multiplier = 1 + (amount / 10000)  # Gradual increase with bet size
        amplitude = min(0.3 + (amount / 5000), 1.0)
        
        return (base_freq * freq_multiplier, amplitude)

    def name_to_marker(self, player_name: str) -> np.ndarray:
        """Generate unique marker sound for each player"""
        # Create a simple identifying tone
        base_freq = hash(player_name) % 500 + 500  # Range: 500-1000 Hz
        return self.generate_tone(base_freq, duration=0.2)

    def round_to_marker(self, round_name: str) -> np.ndarray:
        """Generate round transition marker"""
        # Different rounds get different frequencies
        round_freqs = {
            'Pre-flop': 440,  # A4
            'Flop': 554,      # C#5
            'Turn': 659,      # E5
            'River': 880      # A5
        }
        freq = round_freqs.get(round_name, 440)
        return self.generate_tone(freq, duration=0.5)

    def mix_streams(self) -> np.ndarray:
        """Mix all audio streams into a single output"""
        # First pass: calculate total length needed
        total_duration = 0
        for stream in self.streams.values():
            for sound in stream:
                if isinstance(sound, np.ndarray):
                    total_duration += len(sound)
                elif isinstance(sound, tuple):  # frequency, amplitude pairs
                    total_duration += int(self.DURATION * self.SAMPLE_RATE)
                # Add gap duration
                total_duration += int(self.SAMPLE_RATE * 0.1)  # 100ms gap
        
        # Create output array with proper size
        mixed_audio = np.zeros(total_duration)
        current_position = 0
        
        # Mix each stream type with spacing
        for stream_type, stream in self.streams.items():
            for sound in stream:
                if isinstance(sound, np.ndarray):
                    # Direct audio data
                    end_pos = current_position + len(sound)
                    if end_pos > len(mixed_audio):
                        # Extend mixed_audio if needed
                        mixed_audio = np.pad(mixed_audio, (0, end_pos - len(mixed_audio)))
                    mixed_audio[current_position:end_pos] += sound * 0.5
                    current_position = end_pos
                
                elif isinstance(sound, tuple):
                    # Generate tone from frequency, amplitude pairs
                    freq, amp = sound
                    tone = self.generate_tone(freq) * amp
                    end_pos = current_position + len(tone)
                    if end_pos > len(mixed_audio):
                        # Extend mixed_audio if needed
                        mixed_audio = np.pad(mixed_audio, (0, end_pos - len(mixed_audio)))
                    mixed_audio[current_position:end_pos] += tone * 0.5
                    current_position = end_pos
                
                # Add spacing between sounds
                current_position += int(self.SAMPLE_RATE * 0.1)  # 100ms gap
        
        # Normalize final mix
        if np.max(np.abs(mixed_audio)) > 0:
            mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))
        
        return mixed_audio
