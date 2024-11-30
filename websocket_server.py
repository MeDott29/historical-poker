import asyncio
import websockets
import json
import numpy as np
import logging
from pathlib import Path
import sounddevice as sd
from queue import Queue
import threading
import time
from qwen_gaming_interface import DataChatbot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebAudioPokerServer:
    def __init__(self, host="127.0.0.1", start_port=8765):
        self.host = host
        self.port = start_port
        self.audio_queue = Queue()
        self.running = True
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.input_stream = None
        
        # Create/update index.html with current port
        self.update_index_html()
        logger.info("Updated index.html")
        
        # Start audio interface
        try:
            self.start_audio_capture()
            logger.info("Audio interface started")
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")

    def update_index_html(self):
        """Update the WebSocket URL in index.html with the current port"""
        try:
            with open('index.html', 'r') as file:
                content = file.read()
            
            # Update WebSocket URL
            updated_content = content.replace(
                "const ws = new WebSocket('ws://localhost:[0-9]+')",
                f"const ws = new WebSocket('ws://{self.host}:{self.port}')"
            )
            
            with open('index.html', 'w') as file:
                file.write(updated_content)
            
            logger.info(f"Updated WebSocket URL in index.html to ws://{self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to update index.html: {e}")

    async def websocket_handler(self, websocket, path):
        """Handle websocket connections and stream audio data"""
        try:
            while self.running:
                # Process audio data
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    
                    # Calculate RMS
                    rms = np.sqrt(np.mean(audio_data**2))
                    
                    # Calculate dominant frequency using FFT
                    if len(audio_data) > 0:
                        fft = np.fft.fft(audio_data)
                        freqs = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
                        peak_freq = abs(freqs[np.argmax(np.abs(fft))])
                    else:
                        peak_freq = 0
                    
                    # Create message
                    message = {
                        "rms": float(rms),
                        "freq": float(peak_freq),
                        "action": "ANALYZE"  # Default action
                    }
                    
                    # Send to client
                    await websocket.send(json.dumps(message))
                
                await asyncio.sleep(0.1)  # Prevent busy waiting
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Error in websocket handler: {e}")

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio capture"""
        if status:
            logger.warning(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())

    def start_audio_capture(self):
        """Initialize audio capture"""
        self.input_stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            callback=self.audio_callback
        )
        self.input_stream.start()

    async def start_server(self):
        """Start the websocket server"""
        while True:
            try:
                server = await websockets.serve(
                    self.websocket_handler, 
                    self.host, 
                    self.port,
                    ping_interval=None  # Disable ping/pong for testing
                )
                logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
                
                # Update index.html with the successful port
                self.update_index_html()
                
                await server.wait_closed()
            except OSError as e:
                logger.warning(f"Port {self.port} is in use ({e}), trying next port...")
                self.port += 1
            except Exception as e:
                logger.error(f"Server error: {e}")
                await asyncio.sleep(1)  # Prevent rapid retry on error
            else:
                break

    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()

class WebSocketServer:
    def __init__(self, chatbot: DataChatbot, host: str = "127.0.0.1", port: int = 8766):
        self.chatbot = chatbot
        self.host = host
        self.port = port
        
    async def handle_client(self, websocket, path):
        self.chatbot.ws_clients.add(websocket)
        try:
            while True:
                # Keep connection alive and handle any client messages
                message = await websocket.recv()
                # Process client messages if needed
        except websockets.exceptions.ConnectionClosed:
            self.chatbot.ws_clients.remove(websocket)
            
    async def start(self):
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever

async def main():
    server = WebAudioPokerServer()
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.cleanup()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        logger.info("Server stopped")

if __name__ == "__main__":
    chatbot = DataChatbot(websocket_enabled=True)
    ws_server = WebSocketServer(chatbot)
    
    async def main():
        # Run WebSocket server and chat interface concurrently
        await asyncio.gather(
            ws_server.start(),
            chatbot.chat()
        )
    
    asyncio.run(main()) 