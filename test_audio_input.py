import sounddevice as sd
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def audio_callback(indata, frames, time_info, status):
    if status:
        logger.warning(f"Audio status: {status}")
    
    # Calculate RMS of the audio input
    rms = np.sqrt(np.mean(indata**2))
    logger.info(f"Audio input RMS: {rms:.6f}")

def test_audio_input(duration=10):
    try:
        # Audio stream parameters
        sample_rate = 44100
        channels = 1
        blocksize = 1024

        logger.info("Starting audio input test...")
        logger.info(f"Sample rate: {sample_rate}Hz")
        logger.info(f"Channels: {channels}")
        logger.info(f"Block size: {blocksize}")

        # Create input stream
        with sd.InputStream(
            channels=channels,
            samplerate=sample_rate,
            blocksize=blocksize,
            callback=audio_callback
        ) as stream:
            logger.info("Audio stream opened successfully")
            logger.info(f"Listening for {duration} seconds...")
            time.sleep(duration)

        logger.info("Audio test completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error testing audio input: {e}")
        return False

if __name__ == "__main__":
    test_audio_input() 