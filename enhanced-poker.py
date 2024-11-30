from historical_poker import create_historical_game, PlayerAction
from audio_interface.audio_interface import AudioInterface
from typing import Optional, List, Dict, Iterator
import numpy as np
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import tempfile
import cv2
from pathlib import Path
from video_audio_encoder import VideoAudioEncoder, VideoConfig, AudioConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('poker_game.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PlayerDataManager:
    def __init__(self, data_dir: str = "player_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def save_players(self, players: List[Dict], filename: str = "players.jsonl"):
        path = self.data_dir / filename
        with path.open('w') as f:
            for player in players:
                f.write(json.dumps(player) + '\n')
                
    def load_players(self, filename: str = "players.jsonl") -> Iterator[Dict]:
        path = self.data_dir / filename
        if not path.exists():
            return []
        
        with path.open('r') as f:
            return [json.loads(line) for line in f if line.strip()]
            
    def save_game_results(self, game_id: str, results: Dict):
        path = self.data_dir / f"game_{game_id}_results.jsonl"
        with path.open('a') as f:
            f.write(json.dumps(results) + '\n')

class EnhancedGameManager:
    def __init__(self, batch_size: int = 100):
        self.audio = AudioInterface()
        self.video_encoder = VideoAudioEncoder(
            VideoConfig(fps=30, width=1920, height=1080),
            AudioConfig(sample_rate=44100)
        )
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.player_manager = PlayerDataManager()
        self.batch_size = batch_size
    
    async def process_games(self, game_count: int) -> None:
        """Run multiple poker games in batches"""
        try:
            all_players = self.player_manager.load_players()
            total_batches = (game_count + self.batch_size - 1) // self.batch_size
            
            for batch in range(total_batches):
                batch_players = self._select_players_for_batch(all_players)
                tables = self._create_tables(batch_players)
                
                await asyncio.gather(*[
                    self._run_table(table, f"{batch}_{i}") 
                    for i, table in enumerate(tables)
                ])
                
        except Exception as e:
            logger.error(f"Error processing games: {e}", exc_info=True)

    def _select_players_for_batch(self, all_players: List[Dict]) -> List[List[Dict]]:
        """Select and group players for multiple tables"""
        import random
        players_per_table = random.choice([6, 8, 9])
        batch_players = []
        
        players = random.sample(all_players, min(len(all_players), 
                              self.batch_size * players_per_table))
        
        for i in range(0, len(players), players_per_table):
            table_players = players[i:i + players_per_table]
            if len(table_players) >= 2:  # Minimum players for a game
                batch_players.append(table_players)
        
        return batch_players

    def _create_tables(self, batch_players: List[List[Dict]]):
        """Create multiple game tables"""
        return [create_historical_game(players) for players in batch_players]

    async def _run_table(self, table, game_id: str) -> None:
        """Run a single table's game"""
        try:
            initial_states = self._capture_initial_states(table)
            await self._run_game(table)
            final_states = self._capture_final_states(table)
            
            self.player_manager.save_game_results(game_id, {
                'initial_states': initial_states,
                'final_states': final_states,
                'game_id': game_id
            })
            
        except Exception as e:
            logger.error(f"Error running table {game_id}: {e}")

    def _capture_initial_states(self, table) -> Dict:
        return {
            'players': [{
                'name': p.name,
                'chips': p.chips,
                'persona': p.historical_persona
            } for p in table.players],
            'timestamp': asyncio.get_event_loop().time()
        }

    def _capture_final_states(self, table) -> Dict:
        return {
            'players': [{
                'name': p.name,
                'chips': p.chips,
                'hands_played': getattr(p, 'hands_played', 0),
                'hands_won': getattr(p, 'hands_won', 0)
            } for p in table.players],
            'timestamp': asyncio.get_event_loop().time()
        }

    # [Previous methods remain the same]
    async def _run_game(self, table) -> None:
        """Run poker game rounds"""
        # [Previous implementation remains the same]
        pass

    async def _play_betting_round(self, table, round_name: str, min_bet: int) -> None:
        """Play a single betting round"""
        # [Previous implementation remains the same]
        pass

    def _simulate_action(self, player, min_bet: int) -> tuple[PlayerAction, Optional[int]]:
        """Generate AI player action"""
        # [Previous implementation remains the same]
        pass

    async def _handle_showdown(self, table) -> None:
        """Process game showdown"""
        # [Previous implementation remains the same]
        pass

async def main():
    try:
        # Initialize game manager with larger batch size
        manager = EnhancedGameManager(batch_size=100)
        
        # Generate large player pool
        historical_figures = [
            {"name": "Julius Caesar", "persona": "Roman Emperor", "chips": 2000},
            {"name": "Cleopatra", "persona": "Egyptian Queen", "chips": 2000},
            # Add more historical figures here
        ] * 50  # Multiply to get more players
        
        # Save player data
        manager.player_manager.save_players(historical_figures)
        
        # Run multiple games
        await manager.process_games(game_count=1000)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())