import numpy as np
import sounddevice as sd
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import tkinter as tk
from tkinter import ttk
import threading
import queue
from poker_audio_sim import AudioEncoder
import time
import matplotlib.pyplot as plt
from collections import deque
from scipy.fft import fft, fftfreq

class MultiModalInterface:
    def __init__(self, websocket_mode=False):
        self.audio_queue = queue.Queue()
        self.game_state_queue = queue.Queue()
        self.running = True
        self.websocket_mode = websocket_mode
        self.data_callback = None
        
        if not websocket_mode:
            self.root = tk.Tk()
            self.root.title("Real-time Audio Poker")
            self.setup_gui()
        
        self.audio_encoder = AudioEncoder()
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.input_stream = None
        self.decision_history = deque(maxlen=1000)
        self.audio_buffer = deque(maxlen=44100*10)
        self.last_analysis_time = time.time()
        self.analysis_interval = 2.0

    def setup_gui(self):
        if not self.websocket_mode:
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

            # Real-time display
            self.status_text = tk.Text(main_frame, height=5, width=50)
            self.status_text.grid(row=0, column=0, pady=5)
            
            # Analysis button
            ttk.Button(main_frame, text="Show Analysis", 
                      command=self.show_analysis).grid(row=1, column=0, pady=5)

    def process_audio(self):
        while self.running:
            try:
                audio_data = self.audio_queue.get_nowait()
                self.audio_buffer.extend(audio_data.flatten())
                
                rms = np.sqrt(np.mean(audio_data**2))
                decision = self.make_instant_decision(rms, audio_data)
                
                if decision:
                    self.decision_history.append(decision)
                    
                    if self.websocket_mode and self.data_callback:
                        ws_data = {
                            "timestamp": time.time(),
                            "rms": float(rms),
                            "freq": float(decision["freq"]),
                            "action": decision["action"],
                            "decision_rate": len(self.decision_history) / 
                                          (time.time() - self.start_time)
                        }
                        self.data_callback(ws_data)
                    
            except queue.Empty:
                time.sleep(0.01)
                continue
            except Exception as e:
                if not str(e).startswith("main thread is not in main loop"):
                    logging.error(f"Error in process_audio: {e}")

    def make_instant_decision(self, rms: float, audio_data: np.ndarray) -> Dict:
        # Quick frequency analysis
        if len(audio_data) >= 2:
            freq_data = np.abs(fft(audio_data.flatten())[:len(audio_data)//2])
            dominant_freq = np.argmax(freq_data) * self.sample_rate / len(audio_data)
        else:
            dominant_freq = 0

        # Make decision based on both amplitude and frequency
        action = self.map_noise_to_action(rms, dominant_freq)
        
        return {
            "timestamp": time.time(),
            "action": action,
            "rms": float(rms),
            "freq": float(dominant_freq),
            "raw_audio": audio_data.copy()
        }

    def map_noise_to_action(self, rms: float, freq: float) -> str:
        # Use both amplitude and frequency for decision
        if rms > 0.8 or freq > 2000:
            return "raise"
        elif rms > 0.6 or (freq > 1000 and freq <= 2000):
            return "bet"
        elif rms > 0.4 or (freq > 500 and freq <= 1000):
            return "call"
        else:
            return "fold"

    def trigger_analysis(self):
        self.last_analysis_time = time.time()
        
        # Update status display
        recent_decisions = list(self.decision_history)[-10:]
        status = "Recent decisions:\n"
        for d in recent_decisions:
            status += f"{d['action']} (RMS: {d['rms']:.3f}, Freq: {d['freq']:.1f}Hz)\n"
        
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, status)

    def show_analysis(self):
        if not self.decision_history:
            return

        # Create new figure with subplots
        plt.figure(figsize=(15, 10))

        # Plot 1: Decision distribution
        plt.subplot(2, 2, 1)
        decisions = [d['action'] for d in self.decision_history]
        unique_decisions = list(set(decisions))
        counts = [decisions.count(d) for d in unique_decisions]
        plt.bar(unique_decisions, counts)
        plt.title('Decision Distribution')
        plt.ylabel('Count')

        # Plot 2: RMS over time
        plt.subplot(2, 2, 2)
        rms_values = [d['rms'] for d in self.decision_history]
        plt.plot(rms_values)
        plt.title('RMS Values Over Time')
        plt.ylabel('RMS')

        # Plot 3: Frequency content (FFT of recent audio)
        plt.subplot(2, 2, 3)
        if len(self.audio_buffer) > 1024:
            recent_audio = np.array(list(self.audio_buffer)[-4096:])
            yf = fft(recent_audio)
            xf = fftfreq(len(recent_audio), 1/self.sample_rate)
            plt.plot(xf[:len(xf)//2], 2.0/len(recent_audio) * np.abs(yf[:len(yf)//2]))
            plt.title('Frequency Spectrum')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')

        # Plot 4: Decision timing
        plt.subplot(2, 2, 4)
        decision_times = [d['timestamp'] for d in self.decision_history]
        decision_types = [d['action'] for d in self.decision_history]
        for action in set(decision_types):
            times = [t for t, d in zip(decision_times, decision_types) if d == action]
            y = [1] * len(times)
            plt.scatter(times, y, label=action, alpha=0.5)
        plt.title('Decision Timing')
        plt.ylabel('Occurrence')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def start_audio_capture(self):
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            self.audio_queue.put(indata.copy())

        self.input_stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            callback=audio_callback
        )
        self.input_stream.start()

    def run(self, data_callback=None):
        self.start_time = time.time()
        self.data_callback = data_callback
        self.start_audio_capture()
        
        audio_thread = threading.Thread(target=self.process_audio)
        audio_thread.start()
        
        if not self.websocket_mode:
            try:
                self.root.mainloop()
            finally:
                self.cleanup()
        else:
            # In websocket mode, just keep the audio processing running
            audio_thread.join()

    def cleanup(self):
        self.running = False
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()

if __name__ == "__main__":
    interface = MultiModalInterface()
    interface.run() 