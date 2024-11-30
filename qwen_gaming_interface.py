import ollama
from colorama import init, Fore, Style
from typing import List, Dict
import json
import textwrap
import logging
import random

class DataChatbot:
    cache = {}

    def __init__(self, model: str = "llama3.2:1b", websocket_enabled: bool = False):
        init()  # Initialize colorama
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.client = ollama.Client()
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        
        # WebSocket support
        self.websocket_enabled = websocket_enabled
        self.ws_clients = set()
        
        # Game state tracking
        self.current_game_state = {
            "pot": 0,
            "players": [],
            "current_player": None,
            "player_hand": None,
            "phase": "pregame",  # pregame, dealing, betting, showdown
            "narrative_context": "",
            "ambient_events": []
        }

    def chunk_json(self, json_obj) -> List[str]:
        """More efficient JSON chunking using textwrap."""
        json_str = json.dumps(json_obj, indent=2)
        return textwrap.wrap(json_str, 100, break_long_words=False, break_on_hyphens=False)

    def load_game_data(self) -> Dict:
        if 'game_data' in self.cache:
            return self.cache['game_data']

        data = {
            'games': [],
            'players': [],
            'chunked_games': [],
            'chunked_players': []
        }

        # Load game history
        try:
            with open('data/game_history.jsonl', 'r') as f:
                for line in f:
                    game_data = json.loads(line)
                    data['games'].append(game_data)
                    chunks = self.chunk_json(game_data)
                    data['chunked_games'].append(chunks)
        except Exception as e:
            self.logger.error(f"Error loading game history: {e}")

        # Load player data
        try:
            with open('data/players.jsonl', 'r') as f:
                for line in f:
                    player_data = json.loads(line)
                    data['players'].append(player_data)
                    chunks = self.chunk_json(player_data)
                    data['chunked_players'].append(chunks)
        except Exception as e:
            self.logger.error(f"Error loading player data: {e}")

        self.cache['game_data'] = data
        return data

    def display_chunked_data(self, data_type: str = 'games', index: int = 0):
        chunks = self.game_data[f'chunked_{data_type}'][index]
        print(f"\n{Fore.YELLOW}Displaying {data_type} record {index + 1} in chunks:{Style.RESET_ALL}")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n{Fore.CYAN}Chunk {i}:{Style.RESET_ALL}")
            print(chunk)

    def summarize_data(self, data_type: str, index: int):
        if data_type == 'games':
            game = self.game_data['games'][index]
            summary = (
                f"Game ID: {game.get('id', 'N/A')}, "
                f"Players: {len(game.get('players', []))}, "
                f"Pot Size: {game.get('pot_size', 'N/A')}"
            )
        elif data_type == 'players':
            player = self.game_data['players'][index]
            summary = (
                f"Player Name: {player.get('name', 'N/A')}, "
                f"Wins: {player.get('wins', 'N/A')}, "
                f"Losses: {player.get('losses', 'N/A')}"
            )
        else:
            summary = "Invalid data type"
        return summary

    def generate_response(self, user_input: str) -> str:
        # Create narrative context from game state
        game_context = f"""
        Current Game State:
        - Phase: {self.current_game_state['phase']}
        - Pot: ${self.current_game_state['pot']}
        - Your Hand: {self.current_game_state['player_hand']}
        - Narrative Context: {self.current_game_state['narrative_context']}
        - Recent Events: {', '.join(self.current_game_state['ambient_events'][-3:])}
        """

        prompt = f"""You are a friendly hot dog vendor who watches poker games and tells stories.
        You can:
        - Progress the game state naturally based on user interactions
        - Add ambient events and background details
        - Remember player decisions and their impact
        - Create memorable characters and situations
        - Keep the story engaging and somewhat unpredictable
        
        Game Context: {game_context}
        Recent Conversation: {self.conversation_history[-3:]}
        
        User: {user_input}
        
        Respond naturally as the hot dog vendor, updating the game state and adding color to the scene.
        """

        response = self.client.generate(self.model, prompt=prompt)
        
        # Update game state based on response
        self._update_game_state(user_input, response['response'])
        
        return response['response']

    def _update_game_state(self, user_input: str, ai_response: str):
        # Random ambient events
        ambient_events = [
            "Someone drops their chips",
            "A player orders another drink",
            "The ceiling fan squeaks",
            "Distant laughter from another table",
            "The shuffle machine whirs"
        ]
        
        # Update game state based on context
        if "all in" in user_input.lower():
            self.current_game_state["phase"] = "betting"
            self.current_game_state["pot"] *= 2
        elif "fold" in user_input.lower():
            self.current_game_state["phase"] = "dealing"
            self.current_game_state["pot"] = 0
            
        # Add random ambient detail
        if random.random() < 0.3:  # 30% chance
            self.current_game_state["ambient_events"].append(random.choice(ambient_events))
            
        # Keep only last 5 ambient events
        self.current_game_state["ambient_events"] = self.current_game_state["ambient_events"][-5:]

    def display_hotdog_stand(self):
        stand = f"""
{Fore.YELLOW}
    _    ________________    _
   | |  /                \  | |
   | | |   HOT  DOGS!    | | |
   | | |    ____         | | |
   | | |   |    |        | | |
   | | |   |    | .--.   | | |
   |_| |   |    |'    `  | |_|
       |   |    |        |
     __|___|____|________|__
    /                        \\
   /     🌭 How can I help    \\
  /        you today? 🌭       \\
 /___________________________ \\
{Style.RESET_ALL}"""
        print(stand)

    def chat(self):
        self.display_hotdog_stand()
        print(f"{Fore.CYAN}Welcome to Lucky's Poker Room & Hot Dog Stand! Pull up a chair...{Style.RESET_ALL}")
        
        while True:
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print(f"{Fore.YELLOW}🌭 Thanks for the stories! Come back soon!{Style.RESET_ALL}")
                break

            try:
                self.conversation_history.append({"role": "user", "content": user_input})
                response = self.generate_response(user_input)
                
                # Display any ambient events
                if self.current_game_state["ambient_events"]:
                    latest_event = self.current_game_state["ambient_events"][-1]
                    print(f"{Fore.CYAN}*{latest_event}*{Style.RESET_ALL}")
                
                print(f"{Fore.YELLOW}🌭 Hot Dog Vendor: {response}{Style.RESET_ALL}")
                self.conversation_history.append({"role": "assistant", "content": response})
            except Exception as e:
                self.logger.error(f"Error generating response: {e}")
                print(f"{Fore.RED}Oops! Something went wrong. Can you try again?{Style.RESET_ALL}")

if __name__ == "__main__":
    chatbot = DataChatbot()
    chatbot.chat()