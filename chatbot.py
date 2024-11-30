import ollama
from colorama import init, Fore, Style
from typing import List, Dict
import json
import textwrap

class DataChatbot:
    def __init__(self, model: str = "llama3.2:1b"):
        init()  # Initialize colorama
        self.client = ollama.Client()
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        self.game_data = self.load_game_data()
        
    def chunk_json(self, json_str: str, chunk_size: int = 100) -> List[str]:
        """Split JSON string into chunks of approximately chunk_size characters."""
        return textwrap.wrap(json_str, chunk_size, break_long_words=False, break_on_hyphens=False)
        
    def load_game_data(self) -> Dict:
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
                    # Create chunks of the game data
                    chunks = self.chunk_json(json.dumps(game_data, indent=2))
                    data['chunked_games'].append(chunks)
        except Exception as e:
            print(f"{Fore.RED}Error loading game history: {e}{Style.RESET_ALL}")
            
        # Load player data
        try:
            with open('data/players.jsonl', 'r') as f:
                for line in f:
                    player_data = json.loads(line)
                    data['players'].append(player_data)
                    # Create chunks of the player data
                    chunks = self.chunk_json(json.dumps(player_data, indent=2))
                    data['chunked_players'].append(chunks)
        except Exception as e:
            print(f"{Fore.RED}Error loading player data: {e}{Style.RESET_ALL}")
            
        return data
        
    def display_chunked_data(self, data_type: str = 'games', index: int = 0):
        """Display chunked data with nice formatting."""
        chunks = self.game_data[f'chunked_{data_type}'][index]
        print(f"\n{Fore.YELLOW}Displaying {data_type} record {index + 1} in chunks:{Style.RESET_ALL}")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n{Fore.CYAN}Chunk {i}:{Style.RESET_ALL}")
            print(chunk)
        
    def generate_response(self, user_input: str) -> str:
        # Create context from conversation history and data summary
        context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history[-5:]
        ])
        
        # Add data context with first chunk of first record for each type
        data_context = f"""
        Available game data:
        - {len(self.game_data['games'])} games in history
        - {len(self.game_data['players'])} player records
        
        Sample Game Data (first chunk):
        {self.game_data['chunked_games'][0][0] if self.game_data['chunked_games'] else '{}'}
        
        Sample Player Data (first chunk):
        {self.game_data['chunked_players'][0][0] if self.game_data['chunked_players'] else '{}'}
        """
        
        prompt = f"""You are an AI assistant analyzing poker game data in chunks. 
        Generate a focused response that:
        - Directly answers the user's question about the game data
        - Uses clear and concise language
        - References specific data points when relevant
        - Provides statistical insights when appropriate
        
        Available Data Context:
        {data_context}
        
        Conversation Context:
        {context}
        
        User Question: {user_input}
        """
        
        response = self.client.generate(self.model, prompt=prompt)
        return response['response']
    
    def chat(self):
        print(f"{Fore.CYAN}Welcome to the Game Data Chatbot!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}I can help you analyze the poker game data. Try:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}- 'show game [number]' to see a game record in chunks{Style.RESET_ALL}")
        print(f"{Fore.CYAN}- 'show player [number]' to see a player record in chunks{Style.RESET_ALL}")
        print(f"{Fore.CYAN}- Ask questions about the game data{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Type 'exit' to end the conversation.{Style.RESET_ALL}\n")
        
        while True:
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")
            if user_input.lower() == 'exit':
                break
                
            # Handle special commands
            if user_input.lower().startswith('show game '):
                try:
                    index = int(user_input.split()[-1]) - 1
                    if 0 <= index < len(self.game_data['games']):
                        self.display_chunked_data('games', index)
                    else:
                        print(f"{Fore.RED}Invalid game number. Please choose between 1 and {len(self.game_data['games'])}{Style.RESET_ALL}")
                    continue
                except ValueError:
                    print(f"{Fore.RED}Please provide a valid game number{Style.RESET_ALL}")
                    continue
                    
            if user_input.lower().startswith('show player '):
                try:
                    index = int(user_input.split()[-1]) - 1
                    if 0 <= index < len(self.game_data['players']):
                        self.display_chunked_data('players', index)
                    else:
                        print(f"{Fore.RED}Invalid player number. Please choose between 1 and {len(self.game_data['players'])}{Style.RESET_ALL}")
                    continue
                except ValueError:
                    print(f"{Fore.RED}Please provide a valid player number{Style.RESET_ALL}")
                    continue
            
            # Add user input to history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            try:
                # Generate and display response
                response = self.generate_response(user_input)
                print(f"{Fore.BLUE}Chatbot: {response}{Style.RESET_ALL}")
                
                # Add response to history
                self.conversation_history.append({"role": "assistant", "content": response})
            except Exception as e:
                print(f"{Fore.RED}Error generating response: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    chatbot = DataChatbot()
    chatbot.chat() 