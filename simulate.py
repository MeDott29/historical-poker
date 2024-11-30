import ollama
from colorama import init, Fore, Style
from typing import List, Dict
import json
import textwrap
import logging
import random
import subprocess
import tempfile
import os

class GenericChatbot:
    cache = {}

    def __init__(self, model: str = "llama3.2:1b", websocket_enabled: bool = False, simulation_mode: bool = False):
        init()  # Initialize colorama
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.client = ollama.Client()
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        self.session_state = self._initialize_session_state()
        
        self.websocket_enabled = websocket_enabled
        self.ws_clients = set()
        
        self.simulation_mode = simulation_mode
        self.interactions_log = []
        self.persona_description = "a helpful AI assistant"

    def chunk_json(self, json_obj) -> List[str]:
        json_str = json.dumps(json_obj, indent=2)
        return textwrap.wrap(json_str, 100, break_long_words=False, break_on_hyphens=False)

    def load_context_data(self) -> Dict:
        if 'context_data' in self.cache:
            return self.cache['context_data']

        data = {
            'records': [],
            'chunked_records': []
        }

        try:
            with open('data/context_data.jsonl', 'r') as f:
                for line in f:
                    record = json.loads(line)
                    data['records'].append(record)
                    chunks = self.chunk_json(record)
                    data['chunked_records'].append(chunks)
        except Exception as e:
            self.logger.error(f"Error loading context data: {e}")

        self.cache['context_data'] = data
        return data

    def display_chunked_data(self, index: int):
        chunks = self.context_data['chunked_records'][index]
        print(f"\n{Fore.YELLOW}Displaying record {index + 1} in chunks:{Style.RESET_ALL}")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n{Fore.CYAN}Chunk {i}:{Style.RESET_ALL}")
            print(chunk)

    def summarize_data(self, index: int) -> str:
        record = self.context_data['records'][index]
        summary = f"Record ID: {record.get('id', 'N/A')}, Content: {record.get('content', 'N/A')}"
        return summary

    def generate_response(self, user_input: str) -> str:
        context = self._construct_context()
        
        prompt = f"""
        You are {self.persona_description}.
        Current Scenario: {context}
        Recent Conversation: {self.conversation_history[-3:]}
        
        User: {user_input}
        
        Respond accordingly.
        """
        response = self.client.generate(self.model, prompt=prompt)
        self._update_session_state(user_input, response['response'])
        return response['response']

    def _construct_context(self) -> str:
        return f"Session State: {self.session_state}"

    def _initialize_session_state(self) -> Dict:
        return {
            "key": "value"
            # Customize as needed
        }

    def _update_session_state(self, user_input: str, ai_response: str):
        pass

    def display_greeting(self):
        print(f"{Fore.MAGENTA}Welcome to your chatbot experience!{Style.RESET_ALL}")

    def chat(self):
        if self.simulation_mode:
            self.simulate_chat()
        else:
            self.interactive_chat()

    def simulate_chat(self):
        num_turns = random.randint(1, 2)
        for _ in range(num_turns):
            user_input = self._generate_simulated_user_input()
            self.conversation_history.append({"role": "user", "content": user_input})
            response = self.generate_response(user_input)
            self.conversation_history.append({"role": "assistant", "content": response})
            self.log_interaction(user_input, response)
        self.export_interactions_to_jsonl()
        self.convert_text_to_compressed_audio()

    def _generate_simulated_user_input(self) -> str:
        # Placeholder for generating simulated user input
        return "Hello, how are you?"

    def log_interaction(self, user_input: str, ai_response: str):
        self.interactions_log.append({
            "user_input": user_input,
            "ai_response": ai_response
        })

    def export_interactions_to_jsonl(self):
        with open('simulated_interactions.jsonl', 'w') as f:
            for interaction in self.interactions_log:
                f.write(json.dumps(interaction) + '\n')

    def convert_text_to_compressed_audio(self):
        # Use a temporary file for intermediate WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
            wav_file_path = temp_wav.name
        
        # Convert text to speech using e.g., gtts
        from gtts import gTTS
        text = ' '.join([entry['user_input'] + entry['ai_response'] for entry in self.interactions_log])
        tts = gTTS(text)
        tts.save(wav_file_path)
        
        # Compress WAV to MP3 using ffmpeg
        mp3_file_path = 'simulated_interactions.mp3'
        subprocess.run(['ffmpeg', '-i', wav_file_path, '-vn', '-acodec', 'libmp3lame', '-aq', '2', mp3_file_path])
        
        # Clean up temporary files
        os.remove(wav_file_path)

if __name__ == "__main__":
    simulation_mode = True  # Set to True to enable simulation mode
    chatbot = GenericChatbot(simulation_mode=simulation_mode)
    chatbot.chat()