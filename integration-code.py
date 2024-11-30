from historical_poker import PokerTable, Player, AudioInterface
from typing import List, Dict
import json
import asyncio

class HistoricalDataChatbot(DataChatbot):
    def __init__(self, model: str = "llama3.2:1b"):
        super().__init__(model)
        self.audio_interface = AudioInterface()
        self.table = None

    async def init_game(self):
        """Initialize poker table and audio interface"""
        players_data = self.game_data['players'][:4]  # Get first 4 players
        self.table = PokerTable(
            players=[Player(**p) for p in players_data],
            historical_context={
                "era": "Classical Antiquity",
                "civilization": "Multi-Epochal",
                "architectural_setting": "The Great Library of Alexandria"
            }
        )
        self.table.initialize_deck()

    def record_game_audio(self, game_id: str):
        """Generate and store audio record of game events"""
        game_data = next(g for g in self.game_data['games'] if g['game_id'] == game_id)
        
        # Convert game events to audio patterns
        for event in game_data.get('dramatic_moments', []):
            audio_data = self.audio_interface.generate_tone(
                self.audio_interface.FREQUENCIES[event['type'].lower()]
            )
            self.audio_interface.save_pattern(
                audio_data,
                metadata={"event": event},
                pattern_type="dramatic_moment"
            )

        # Save complete game audio
        self.audio_interface.save_buffer(f"game_{game_id}_audio.wav")

    async def analyze_game_patterns(self, game_id: str):
        """Analyze audio patterns from a specific game"""
        audio_data = await self.audio_interface.load_audio(f"game_{game_id}_audio.wav")
        patterns = self.audio_interface.find_similar_patterns(
            audio_data,
            pattern_type="dramatic_moment",
            threshold=0.8
        )
        return patterns

    def generate_response(self, user_input: str) -> str:
        """Enhanced response generation with audio context"""
        audio_context = f"""
        Audio patterns available:
        - Card sounds based on suit/value
        - Betting sounds scaled by amount
        - Special event markers
        - Historical commentary audio
        """
        
        # Add audio context to prompt
        prompt = super().generate_response(user_input)
        return prompt + "\n\nAudio Context:\n" + audio_context

async def main():
    chatbot = HistoricalDataChatbot()
    await chatbot.init_game()
    chatbot.chat()