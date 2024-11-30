import asyncio
import numpy as np
import soundfile as sf
import cv2
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from dataclasses import dataclass, field
import time
from statistics import mean

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

@dataclass
class PipelineMetrics:
    encoding_times: List[float] = field(default_factory=list)
    decoding_times: List[float] = field(default_factory=list)
    audio_processing_times: List[float] = field(default_factory=list)
    visualization_times: List[float] = field(default_factory=list)
    success_rate: float = 0.0
    total_data_processed: int = 0
    failed_operations: Dict[str, int] = field(default_factory=lambda: {
        "encoding": 0,
        "decoding": 0,
        "audio": 0,
        "visualization": 0
    })

    def summarize(self) -> Dict:
        return {
            "avg_encoding_time": mean(self.encoding_times) if self.encoding_times else 0,
            "avg_decoding_time": mean(self.decoding_times) if self.decoding_times else 0,
            "avg_audio_processing_time": mean(self.audio_processing_times) if self.audio_processing_times else 0,
            "avg_visualization_time": mean(self.visualization_times) if self.visualization_times else 0,
            "success_rate": self.success_rate,
            "total_processed": self.total_data_processed,
            "failed_operations": self.failed_operations
        }

class DataEncoder:
    def __init__(self, config: DecoderConfig = None):
        self.config = config or DecoderConfig()
        self.metrics = PipelineMetrics()

    @staticmethod
    def timer_decorator(metric_list):
        def decorator(func):
            async def wrapper(self, *args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = await func(self, *args, **kwargs) if asyncio.iscoroutinefunction(func) else func(self, *args, **kwargs)
                    getattr(self.metrics, metric_list).append(time.perf_counter() - start_time)
                    return result
                except Exception as e:
                    logger.error(f"Error in {func.__name__}: {e}")
                    self.metrics.failed_operations[metric_list.replace("_times", "")] += 1
                    return None
            return wrapper if asyncio.iscoroutinefunction(func) else func
        return decorator

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

    @timer_decorator(metric_list="encoding_times")
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
        self.metrics = PipelineMetrics()
        self.logger = logging.getLogger(__name__)

    @DataEncoder.timer_decorator(metric_list="audio_processing_times")
    def load_audio(self, file_path: str) -> Optional[Tuple[np.ndarray, int]]:
        """Load audio file and return signal and sample rate."""
        try:
            audio_signal, sample_rate = sf.read(file_path)
            self.logger.info(f"Loaded audio file: {file_path}")
            return audio_signal, sample_rate
        except Exception as e:
            self.logger.error(f"Error loading audio file: {str(e)}")
            return None

    @DataEncoder.timer_decorator(metric_list="decoding_times")
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

    @DataEncoder.timer_decorator(metric_list="visualization_times")
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

async def test_different_player_counts():
    simulator = PlayerSimulator()
    encoder = DataEncoder()
    decoder = AudioVideoDecoder()

    # Test with 2 players
    players = await simulator.generate_player_pool(players_per_era=2)
    matchup = await simulator.get_matchup(num_players=2)
    data_to_encode = matchup[0]
    brightness_values = encoder.normalize_data(data_to_encode)
    frequencies = encoder.brightness_to_frequencies(brightness_values)
    audio_signal = encoder.generate_audio_signal(frequencies)
    audio_file = Path("output") / "encoded_data_2_players.wav"
    sf.write(str(audio_file), audio_signal, 44100)
    success = decoder.decode(str(audio_file))
    assert success, "Decoding failed for 2 players"

    # Test with 4 players
    players = await simulator.generate_player_pool(players_per_era=4)
    matchup = await simulator.get_matchup(num_players=4)
    data_to_encode = matchup[0]
    brightness_values = encoder.normalize_data(data_to_encode)
    frequencies = encoder.brightness_to_frequencies(brightness_values)
    audio_signal = encoder.generate_audio_signal(frequencies)
    audio_file = Path("output") / "encoded_data_4_players.wav"
    sf.write(str(audio_file), audio_signal, 44100)
    success = decoder.decode(str(audio_file))
    assert success, "Decoding failed for 4 players"

async def test_different_player_stats():
    simulator = PlayerSimulator()
    encoder = DataEncoder()
    decoder = AudioVideoDecoder()

    # Simulate player data with different types of stats
    players = [
        {'name': 'Player_1', 'stats': {'score': 95, 'level': 10, 'status': 'active'}},
        {'name': 'Player_2', 'stats': {'score': 85, 'level': 8, 'status': 'inactive'}},
        {'name': 'Player_3', 'stats': {'score': 75, 'level': 6, 'status': 'active'}},
        {'name': 'Player_4', 'stats': {'score': 65, 'level': 4, 'status': 'inactive'}}
    ]

    data_to_encode = players[0]
    brightness_values = encoder.normalize_data(data_to_encode)
    frequencies = encoder.brightness_to_frequencies(brightness_values)
    audio_signal = encoder.generate_audio_signal(frequencies)
    audio_file = Path("output") / "encoded_data_different_stats.wav"
    sf.write(str(audio_file), audio_signal, 44100)
    success = decoder.decode(str(audio_file))
    assert success, "Decoding failed for different player stats"

async def test_different_sample_rates():
    simulator = PlayerSimulator()
    encoder = DataEncoder()
    decoder = AudioVideoDecoder()

    # Test with sample rate 22050
    players = await simulator.generate_player_pool(players_per_era=2)
    matchup = await simulator.get_matchup(num_players=2)
    data_to_encode = matchup[0]
    brightness_values = encoder.normalize_data(data_to_encode)
    frequencies = encoder.brightness_to_frequencies(brightness_values)
    audio_signal = encoder.generate_audio_signal(frequencies, sample_rate=22050)
    audio_file = Path("output") / "encoded_data_22050_sample_rate.wav"
    sf.write(str(audio_file), audio_signal, 22050)
    success = decoder.decode(str(audio_file))
    assert success, "Decoding failed for sample rate 22050"

    # Test with sample rate 48000
    audio_signal = encoder.generate_audio_signal(frequencies, sample_rate=48000)
    audio_file = Path("output") / "encoded_data_48000_sample_rate.wav"
    sf.write(str(audio_file), audio_signal, 48000)
    success = decoder.decode(str(audio_file))
    assert success, "Decoding failed for sample rate 48000"

async def test_different_fft_parameters():
    config = DecoderConfig(window_size=1024, hop_length=512)
    simulator = PlayerSimulator()
    encoder = DataEncoder(config=config)
    decoder = AudioVideoDecoder(config=config)

    players = await simulator.generate_player_pool(players_per_era=2)
    matchup = await simulator.get_matchup(num_players=2)
    data_to_encode = matchup[0]
    brightness_values = encoder.normalize_data(data_to_encode)
    frequencies = encoder.brightness_to_frequencies(brightness_values)
    audio_signal = encoder.generate_audio_signal(frequencies)
    audio_file = Path("output") / "encoded_data_custom_fft_params.wav"
    sf.write(str(audio_file), audio_signal, 44100)
    success = decoder.decode(str(audio_file))
    assert success, "Decoding failed for custom FFT parameters"

async def test_different_video_dimensions():
    config = DecoderConfig(width=800, height=600)
    encoder = DataEncoder(config=config)
    decoder = AudioVideoDecoder(config=config)

    players = await simulator.generate_player_pool(players_per_era=2)
    matchup = await simulator.get_matchup(num_players=2)
    data_to_encode = matchup[0]
    brightness_values = encoder.normalize_data(data_to_encode)
    frequencies = encoder.brightness_to_frequencies(brightness_values)
    audio_signal = encoder.generate_audio_signal(frequencies)
    audio_file = Path("output") / "encoded_data_custom_video_dimensions.wav"
    sf.write(str(audio_file), audio_signal, 44100)
    success = decoder.decode(str(audio_file))
    assert success, "Decoding failed for custom video dimensions"

async def test_different_audio_files():
    decoder = AudioVideoDecoder()

    # Test with a different audio file
    audio_file = "path_to_another_audio_file.wav"
    success = decoder.decode(audio_file)
    assert success, f"Decoding failed for audio file: {audio_file}"

    # Test with a non-existent audio file
    audio_file = "non_existent_audio_file.wav"
    success = decoder.decode(audio_file)
    assert not success, f"Decoding succeeded for non-existent audio file: {audio_file}"

async def main():
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize metrics collector
    metrics_log = []
    
    try:
        # Initialize components
        simulator = PlayerSimulator()
        encoder = DataEncoder()
        decoder = AudioVideoDecoder()

        # Process multiple batches
        num_batches = 5
        for batch in range(num_batches):
            logger.info(f"Processing batch {batch + 1}/{num_batches}")
            
            # Generate and process test data
            players = await simulator.generate_player_pool(players_per_era=2)
            matchup = await simulator.get_matchup(num_players=4)

            start_time = time.perf_counter()
            
            # Process batch
            data_to_encode = matchup[0]
            brightness_values = encoder.normalize_data(data_to_encode)
            frequencies = encoder.brightness_to_frequencies(brightness_values)
            audio_signal = encoder.generate_audio_signal(frequencies)
            
            # Save and decode
            audio_file = output_dir / f"encoded_data_batch_{batch}.wav"
            sf.write(str(audio_file), audio_signal, 44100)
            success = decoder.decode(str(audio_file))
            
            # Update metrics
            batch_time = time.perf_counter() - start_time
            encoder.metrics.total_data_processed += len(data_to_encode)
            encoder.metrics.success_rate = (
                sum(1 for f in audio_file.parent.glob("encoded_data_batch_*.wav")) / (batch + 1)
            )
            
            # Log batch metrics
            metrics_log.append({
                "batch": batch,
                "processing_time": batch_time,
                "encoder_metrics": encoder.metrics.summarize(),
                "decoder_metrics": decoder.metrics.summarize()
            })

        # Log final metrics summary
        logger.info("Pipeline Performance Metrics:")
        for metric in metrics_log:
            logger.info(f"Batch {metric['batch']}: {metric['processing_time']:.2f}s")
            logger.info(f"Encoder metrics: {metric['encoder_metrics']}")
            logger.info(f"Decoder metrics: {metric['decoder_metrics']}")

        # Save metrics to file
        metrics_file = output_dir / "pipeline_metrics.json"
        with metrics_file.open('w') as f:
            json.dump(metrics_log, f, indent=2)

    except Exception as e:
        logger.error(f"Error in main process: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())