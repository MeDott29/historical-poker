import cv2
import numpy as np
import soundfile as sf
import scipy.signal as signal
import datetime
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class VideoConfig:
    fps: int = 24
    width: int = 640
    height: int = 480
    max_frames: int = 300  # Limits recording to ~12.5 seconds at 24 FPS

@dataclass
class AudioConfig:
    sample_rate: int = 44100
    duration_per_frame: float = 0.1
    min_frequency: float = 440.0
    max_frequency: float = 940.0

class VideoAudioEncoder:
    def __init__(self, video_config: VideoConfig = None, audio_config: AudioConfig = None):
        """Initialize the encoder with configurable parameters."""
        self.video_config = video_config or VideoConfig()
        self.audio_config = audio_config or AudioConfig()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('encoder.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize run records
        self.run_records = []
        self.load_run_records()

    def load_run_records(self):
        """Load run records from the log file."""
        try:
            with open('encoder.log', 'r') as log_file:
                lines = log_file.readlines()
                for line in lines:
                    if "Run started at" in line:
                        start_time = line.split("at ")[1].strip()
                    elif "Run ended at" in line:
                        end_time = line.split("at ")[1].strip()
                        duration = self.calculate_duration(start_time, end_time)
                        self.run_records.append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': duration
                        })
        except Exception as e:
            self.logger.error(f"Error loading run records: {str(e)}")

    @staticmethod
    def calculate_duration(start_time_str, end_time_str):
        """Calculate duration between two time strings."""
        start_time = datetime.datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S,%f')
        end_time = datetime.datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S,%f')
        return (end_time - start_time).total_seconds()

    def get_average_run_time(self):
        """Calculate the average run time based on historical records."""
        if not self.run_records:
            return None
        total_duration = sum(record['duration'] for record in self.run_records)
        average_duration = total_duration / len(self.run_records)
        return average_duration

    def capture_video(self) -> Optional[List[np.ndarray]]:
        """
        Capture video from webcam until 'q' is pressed or max frames reached.
        Returns list of frames or None if capture fails.
        """
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not open webcam")

            # Configure webcam
            cap.set(cv2.CAP_PROP_FPS, self.video_config.fps)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_config.height)

            frames = []
            start_time = time.time()

            while len(frames) < self.video_config.max_frames:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Failed to read frame from webcam")
                    break

                frames.append(frame)
                
                # Display frame with recording information
                frame_count_text = f"Recording: {len(frames)} frames"
                cv2.putText(frame, frame_count_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Webcam (Press q to stop)', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            capture_time = time.time() - start_time
            self.logger.info(f"Captured {len(frames)} frames in {capture_time:.2f} seconds")
            
            return frames

        except Exception as e:
            self.logger.error(f"Error during video capture: {str(e)}")
            return None

        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()

    def process_frames(self, frames: List[np.ndarray]) -> Optional[List[float]]:
        """
        Extract features (brightness) from frames.
        Returns list of feature values or None if processing fails.
        """
        try:
            start_time = time.time()
            features = []

            for i, frame in enumerate(frames):
                # Convert to grayscale and calculate average brightness
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                avg_brightness = np.mean(gray)
                features.append(avg_brightness)

                if (i + 1) % 10 == 0:  # Log progress every 10 frames
                    self.logger.debug(f"Processed {i + 1}/{len(frames)} frames")

            processing_time = time.time() - start_time
            self.logger.info(f"Processed {len(frames)} frames in {processing_time:.2f} seconds")
            
            return features

        except Exception as e:
            self.logger.error(f"Error during frame processing: {str(e)}")
            return None

    def encode_audio(self, features: List[float]) -> Optional[Tuple[np.ndarray, int]]:
        """
        Convert features to audio signal.
        Returns tuple of (audio_signal, sample_rate) or None if encoding fails.
        """
        try:
            start_time = time.time()
            num_frames = len(features)
            total_samples = int(num_frames * self.audio_config.sample_rate * 
                              self.audio_config.duration_per_frame)
            audio_signal = np.zeros(total_samples)

            for i, brightness in enumerate(features):
                # Map brightness (0-255) to frequency range
                freq = (self.audio_config.min_frequency + 
                       (brightness / 255.0) * 
                       (self.audio_config.max_frequency - self.audio_config.min_frequency))
                
                # Generate time array for this frame
                t = np.linspace(
                    0, 
                    self.audio_config.duration_per_frame,
                    int(self.audio_config.sample_rate * self.audio_config.duration_per_frame),
                    False
                )
                
                # Generate audio cycle and apply smooth envelope
                audio_cyc = np.sin(freq * t * 2 * np.pi)
                envelope = np.hanning(len(audio_cyc))
                audio_cyc *= envelope
                
                # Insert into main audio signal
                start_sample = int(i * self.audio_config.sample_rate * 
                                 self.audio_config.duration_per_frame)
                end_sample = start_sample + len(audio_cyc)
                audio_signal[start_sample:end_sample] = audio_cyc

            encoding_time = time.time() - start_time
            self.logger.info(f"Encoded audio in {encoding_time:.2f} seconds")
            
            return audio_signal, self.audio_config.sample_rate

        except Exception as e:
            self.logger.error(f"Error during audio encoding: {str(e)}")
            return None

    def save_audio(self, audio_data: Tuple[np.ndarray, int], 
                  output_dir: str = "output") -> Optional[str]:
        """
        Save audio data to WAV file.
        Returns path to saved file or None if saving fails.
        """
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_path / f"encoded_video_{timestamp}.wav"

            # Save audio file
            audio_signal, sample_rate = audio_data
            sf.write(filename, audio_signal, sample_rate)
            
            self.logger.info(f"Saved audio file: {filename}")
            return str(filename)

        except Exception as e:
            self.logger.error(f"Error saving audio file: {str(e)}")
            return None

    def run(self) -> bool:
        """
        Execute the complete video-to-audio encoding process.
        Returns True if successful, False otherwise.
        """
        self.logger.info("Starting video-to-audio encoding process")
        
        # Record start time
        start_time = time.time()
        self.logger.info(f"Run started at {datetime.datetime.now()}")

        # Capture video
        self.logger.info("Starting video capture (press 'q' to stop)...")
        frames = self.capture_video()
        if frames is None or len(frames) == 0:
            self.logger.error("Video capture failed or no frames captured")
            return False

        # Process frames
        self.logger.info("Processing video frames...")
        features = self.process_frames(frames)
        if features is None:
            self.logger.error("Frame processing failed")
            return False

        # Encode audio
        self.logger.info("Encoding audio...")
        audio_data = self.encode_audio(features)
        if audio_data is None:
            self.logger.error("Audio encoding failed")
            return False

        # Save audio file
        self.logger.info("Saving audio file...")
        output_file = self.save_audio(audio_data)
        if output_file is None:
            self.logger.error("Failed to save audio file")
            return False

        # Record end time and duration
        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f"Run ended at {datetime.datetime.now()}, Duration: {duration:.2f} seconds")
        
        self.run_records.append({
            'start_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f'),
            'end_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f'),
            'duration': duration
        })

        self.logger.info("Video-to-audio encoding completed successfully")
        return True

def main():
    """Main entry point for the application."""
    encoder = VideoAudioEncoder()
    success = encoder.run()
    
    if not success:
        logging.error("Encoding process failed")
        exit(1)
    
    logging.info("Process completed successfully")

if __name__ == "__main__":
    main()