import numpy as np
import soundfile as sf
import json
import time
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioGameEncoder:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.f0 = 1000  # Frequency for bit '0'
        self.f1 = 2000  # Frequency for bit '1'
        self.carrier_freq = 440  # Not used in BFSK
        self.bit_duration = 0.01  # Duration of each bit in seconds
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

    def bytes_to_bits(self, byte_data: bytes) -> np.ndarray:
        """Convert bytes to a numpy array of bits"""
        bits = np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8))
        return bits

    def bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """Convert a numpy array of bits to bytes"""
        bytes_array = np.packbits(bits)
        return bytes_array.tobytes()

    def encode_frame(self, game_data: Dict[str, Any]) -> np.ndarray:
        """Encode a single frame of game data into an audio signal using BFSK"""
        json_data = json.dumps(game_data)
        byte_data = json_data.encode('utf-8')
        bits = self.bytes_to_bits(byte_data)
        
        # Calculate frame duration based on number of bits
        frame_duration = len(bits) * self.bit_duration
        
        # Generate the signal for each bit
        signal = np.array([], dtype=np.float32)
        for bit in bits:
            t = np.linspace(0, self.bit_duration, int(self.sample_rate * self.bit_duration), endpoint=False)
            freq = self.f1 if bit else self.f0
            s = np.sin(2 * np.pi * freq * t)
            signal = np.concatenate((signal, s))
        
        return signal

    def decode_frame(self, audio_signal: np.ndarray) -> Dict[str, Any]:
        """Decode an audio signal back to game data using BFSK demodulation"""
        try:
            samples_per_bit = int(self.sample_rate * self.bit_duration)
            num_bits = len(audio_signal) // samples_per_bit
            bits = []
            for i in range(num_bits):
                # Extract the chunk for the current bit
                start = i * samples_per_bit
                end = start + samples_per_bit
                chunk = audio_signal[start:end]
                
                # Perform FFT to find the dominant frequency
                freqs = np.fft.fftfreq(len(chunk), d=1/self.sample_rate)
                fft_magnitude = np.abs(np.fft.fft(chunk))
                positive_freqs = freqs[:len(freqs)//2]
                positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]
                dominant_freq = positive_freqs[np.argmax(positive_magnitude)]
                
                # Determine if the bit is '0' or '1'
                if abs(dominant_freq - self.f0) < abs(dominant_freq - self.f1):
                    bits.append(0)
                else:
                    bits.append(1)
            
            bits = np.array(bits, dtype=np.uint8)
            byte_data = self.bits_to_bytes(bits)
            json_data = byte_data.decode('utf-8')
            return json.loads(json_data)
        except Exception as e:
            logger.error(f"Error decoding frame: {e}")
            return {}

    def update_visualization(self, frame_num: int, audio_signal: np.ndarray, game_data: Dict[str, Any]):
        """Update the real-time visualization"""
        self.ax1.clear()
        self.ax2.clear()

        # Plot audio waveform
        t = np.linspace(0, len(audio_signal)/self.sample_rate, len(audio_signal))
        self.ax1.plot(t, audio_signal)
        self.ax1.set_title(f'Audio Signal - Frame {frame_num + 1}')
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Amplitude')

        # Plot game state
        self.ax2.bar(['Score', 'Health'], 
                     [game_data.get('score', 0), game_data.get('health', 0)])
        self.ax2.set_title('Game State')
        self.ax2.set_ylim(0, 100)

        plt.tight_layout()
        plt.pause(0.01)

    def run_simulation(self, num_frames: int = 50):
        """Run the full simulation with encoding, decoding, and visualization"""
        all_audio = np.array([], dtype=np.float32)
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