import asyncio
import numpy as np
import soundfile as sf
import cv2
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('combined_script.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DecoderConfig:
    width: int = 640
    height: int = 480
    min_frequency: float = 440.0
    max_frequency: float = 940.0
    window_size: int = 2048  # For FFT analysis
    hop_length: int = 1024   # For overlapping windows

class DataEncoder:
    def __init__(self, config: DecoderConfig = None):
        self.config = config or DecoderConfig()

    def brightness_to_frequencies(self, brightness_values: List[float]) -> List[float]:
        """Convert brightness values to frequencies"""
        freq_range = self.config.max_frequency - self.config.min_frequency
        return [
            self.config.min_frequency + (brightness / 255.0) * freq_range
            for brightness in brightness_values
        ]

    def generate_audio_signal(self, frequencies: List[float], sample_rate: int = 44100) -> np.ndarray:
        """Generate audio signal from frequencies"""
        duration = 0.1  # seconds per value
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        signal = np.array([])
        for freq in frequencies:
            tone = np.sin(2 * np.pi * freq * t)
            signal = np.concatenate([signal, tone])
        # Normalize the signal to prevent clipping
        signal = signal / np.max(np.abs(signal))
        return signal

    def normalize_data(self, data: Dict) -> List[float]:
        """Convert dictionary data to normalized brightness values"""
        # Convert dict to JSON string and then to bytes
        json_str = json.dumps(data)
        data_bytes = json_str.encode('utf-8')
        # Convert bytes to brightness values (0-255)
        return [byte for byte in data_bytes]

class AudioVideoDecoder:
    def __init__(self, config: DecoderConfig = None):
        self.config = config or DecoderConfig()
        self.logger = logging.getLogger(__name__)

    def load_audio(self, file_path: str) -> Optional[Tuple[np.ndarray, int]]:
        """Load audio file and return signal and sample rate."""
        try:
            audio_signal, sample_rate = sf.read(file_path)
            self.logger.info(f"Loaded audio file: {file_path}")
            return audio_signal, sample_rate
        except Exception as e:
            self.logger.error(f"Error loading audio file: {str(e)}")
            return None

    def extract_frequencies(self, audio_data: Tuple[np.ndarray, int]) -> Optional[List[float]]:
        """Extract dominant frequencies from audio segments."""
        try:
            audio_signal, sample_rate = audio_data
            frequencies = []
            num_samples = len(audio_signal)
            for i in range(0, num_samples - self.config.window_size, self.config.hop_length):
                # Extract window
                window = audio_signal[i:i + self.config.window_size]
                # Apply Hanning window
                window = window * np.hanning(len(window))
                # Compute FFT
                fft = np.fft.rfft(window)
                freqs = np.fft.rfftfreq(len(window), 1 / sample_rate)
                # Find dominant frequency in our range of interest
                mask = (freqs >= self.config.min_frequency) & (freqs <= self.config.max_frequency)
                if not any(mask):
                    continue
                magnitude = np.abs(fft[mask])
                if len(magnitude) == 0:
                    continue
                dominant_freq = freqs[mask][np.argmax(magnitude)]
                frequencies.append(dominant_freq)
            self.logger.info(f"Extracted {len(frequencies)} frequency values")
            return frequencies
        except Exception as e:
            self.logger.error(f"Error extracting frequencies: {str(e)}")
            return None

    def frequencies_to_brightness(self, frequencies: List[float]) -> List[float]:
        """Convert frequencies back to brightness values."""
        freq_range = self.config.max_frequency - self.config.min_frequency
        return [
            255 * (freq - self.config.min_frequency) / freq_range
            for freq in frequencies
        ]

    def create_visualization(self, brightness_values: List[float]) -> None:
        """Create visualization window showing the decoded brightness values."""
        try:
            # Create window
            cv2.namedWindow('Decoded Video', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Decoded Video', self.config.width, self.config.height)
            for brightness in brightness_values:
                # Clip brightness value to [0,255]
                brightness = np.clip(brightness, 0, 255)
                # Create frame with the decoded brightness
                frame = np.full((self.config.height, self.config.width), brightness, dtype=np.uint8)
                # Display frame
                cv2.imshow('Decoded Video', frame)
                # Wait for 100ms or key press
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
        except Exception as e:
            self.logger.error(f"Error during visualization: {str(e)}")

    def decode(self, audio_file: str) -> bool:
        """Main decoding process."""
        try:
            self.logger.info(f"Starting decoding process for {audio_file}")
            # Load audio
            audio_data = self.load_audio(audio_file)
            if audio_data is None:
                return False
            # Extract frequencies
            frequencies = self.extract_frequencies(audio_data)
            if frequencies is None:
                return False
            # Convert to brightness values
            brightness_values = self.frequencies_to_brightness(frequencies)
            # Visualize
            self.create_visualization(brightness_values)
            self.logger.info("Decoding completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Decoding failed: {str(e)}")
            return False

class PlayerSimulator:
    async def generate_player_pool(self, players_per_era=2):
        # Simulate generating a pool of players
        # For demonstration purposes, we return a list of dummy players
        return [
            {'name': f'Player_{i+1}', 'stats': {'score': np.random.randint(0, 100)}}
            for i in range(players_per_era * 5)
        ]

    async def get_matchup(self, num_players=4):
        # Simulate getting a matchup among the players
        players = await self.generate_player_pool(players_per_era=num_players)
        # For simplicity, return the first num_players players
        return players[:num_players]

async def main():
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    try:
        # Initialize components
        simulator = PlayerSimulator()
        encoder = DataEncoder()
        decoder = AudioVideoDecoder()

        # Generate test data
        players = await simulator.generate_player_pool(players_per_era=2)
        matchup = await simulator.get_matchup(num_players=4)

        # Encode data
        logger.info("Encoding player data to brightness values...")
        # For demonstration, let's encode the first player from the matchup
        data_to_encode = matchup[0]
        brightness_values = encoder.normalize_data(data_to_encode)

        # Convert to frequencies
        frequencies = encoder.brightness_to_frequencies(brightness_values)
        logger.info(f"Converted brightness values to frequencies")

        # Generate audio signal
        audio_signal = encoder.generate_audio_signal(frequencies)
        logger.info(f"Generated audio signal from frequencies")

        # Save audio file
        audio_file = output_dir / "encoded_data.wav"
        sf.write(str(audio_file), audio_signal, 44100)
        logger.info(f"Saved encoded audio to {audio_file}")

        # Decode audio file
        logger.info("Decoding audio file...")
        success = decoder.decode(str(audio_file))
        if success:
            logger.info("Successfully encoded and decoded data")
        else:
            logger.error("Failed to decode data")

    except Exception as e:
        logger.error(f"Error in main process: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())