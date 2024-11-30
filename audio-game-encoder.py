import numpy as np
import soundfile as sf
import json
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioGameEncoder:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.carrier_freq = 440  # Base frequency for modulation
        self.frame_duration = 0.1  # Duration per game state frame
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up visualization
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plt.ion()
        
    def simulate_gameplay(self) -> Dict[str, Any]:
        """Simulate a game state"""
        return {
            'player_position': [np.random.rand(), np.random.rand()],
            'score': np.random.randint(0, 100),
            'health': np.random.randint(0, 100),
            'event': np.random.choice(['collision', 'goal', 'power_up', 'none']),
            'timestamp': time.time()
        }

    def encode_frame(self, game_data: Dict[str, Any]) -> np.ndarray:
        """Encode a single frame of game data into audio"""
        json_data = json.dumps(game_data)
        byte_data = json_data.encode('utf-8')
        data_points = np.array(list(byte_data), dtype=float)
        
        # Normalize data to [-1, 1]
        normalized_data = 2 * (data_points - data_points.min()) / (data_points.max() - data_points.min()) - 1
        
        # Generate time array for this frame
        t = np.linspace(0, self.frame_duration, int(self.sample_rate * self.frame_duration))
        
        # Modulate the carrier wave with game data
        signal = normalized_data.mean() * np.sin(2 * np.pi * self.carrier_freq * t)
        
        return signal

    def decode_frame(self, audio_signal: np.ndarray) -> Dict[str, Any]:
        """Decode audio signal back to game data"""
        try:
            # Demodulate signal
            t = np.linspace(0, self.frame_duration, len(audio_signal))
            demodulated = audio_signal / np.sin(2 * np.pi * self.carrier_freq * t)
            
            # Convert back to bytes and parse JSON
            avg_value = np.mean(demodulated)
            normalized = (avg_value + 1) / 2  # Convert back from [-1, 1] to [0, 1]
            
            # Reconstruct original byte range (0-255)
            byte_value = int(normalized * 255)
            decoded_data = bytes([byte_value])
            
            return json.loads(decoded_data.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error decoding frame: {e}")
            return {}

    def update_visualization(self, frame_num: int, audio_signal: np.ndarray, game_data: Dict[str, Any]):
        """Update the real-time visualization"""
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot audio waveform
        t = np.linspace(0, self.frame_duration, len(audio_signal))
        self.ax1.plot(t, audio_signal)
        self.ax1.set_title('Audio Signal')
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Amplitude')
        
        # Plot game state
        self.ax2.bar(['Score', 'Health'], 
                    [game_data.get('score', 0), game_data.get('health', 0)])
        self.ax2.set_title('Game State')
        self.ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.pause(0.01)

    def run_simulation(self, num_frames: int = 100):
        """Run the full simulation with encoding, decoding, and visualization"""
        all_audio = np.array([])
        original_data = []
        decoded_data = []
        
        for frame in range(num_frames):
            # Generate and encode game data
            game_data = self.simulate_gameplay()
            original_data.append(game_data)
            
            audio_frame = self.encode_frame(game_data)
            all_audio = np.concatenate([all_audio, audio_frame])
            
            # Decode and verify
            decoded_frame = self.decode_frame(audio_frame)
            decoded_data.append(decoded_frame)
            
            # Update visualization
            self.update_visualization(frame, audio_frame, game_data)
            
            logger.info(f"Frame {frame + 1}/{num_frames} processed")
        
        # Save the complete audio file
        sf.write(self.output_dir / 'game_audio.wav', all_audio, self.sample_rate)
        
        # Save original and decoded data for comparison
        with open(self.output_dir / 'original_data.json', 'w') as f:
            json.dump(original_data, f, indent=2)
        with open(self.output_dir / 'decoded_data.json', 'w') as f:
            json.dump(decoded_data, f, indent=2)
        
        return original_data, decoded_data

def compare_data(original: list, decoded: list) -> float:
    """Compare original and decoded data, return similarity score"""
    total_fields = 0
    matched_fields = 0
    
    for orig, dec in zip(original, decoded):
        for key in orig:
            total_fields += 1
            if key in dec and orig[key] == dec[key]:
                matched_fields += 1
    
    return matched_fields / total_fields if total_fields > 0 else 0.0

def main():
    encoder = AudioGameEncoder()
    original, decoded = encoder.run_simulation(num_frames=50)
    
    similarity = compare_data(original, decoded)
    logger.info(f"Data integrity score: {similarity:.2%}")
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()