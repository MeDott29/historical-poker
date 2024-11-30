from historical_poker import create_historical_game, PlayerAction
from audio_interface.audio_interface import AudioInterface
from typing import Optional, List, Dict
import numpy as np
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
import cv2
from video_audio_encoder import VideoAudioEncoder, VideoConfig, AudioConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('poker_game.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedGameManager:
    def __init__(self):
        self.audio = AudioInterface()
        self.video_encoder = VideoAudioEncoder(
            VideoConfig(fps=30, width=640, height=480),
            AudioConfig(sample_rate=44100)
        )
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def process_game(self, players_data: List[Dict]) -> None:
        """Run a complete poker game with video processing"""
        try:
            # Initialize game components
            table = create_historical_game(players_data)
            self.audio.initialize_streams()
            
            # Start video capture and processing
            frames = await self._capture_game_video(table)
            if frames:
                features = self.video_encoder.process_frames(frames)
                if features:
                    audio_data = self.video_encoder.encode_audio(features)
                    if audio_data:
                        self.video_encoder.save_audio(audio_data, "game_recordings")
            
            # Run the game
            await self._run_game(table)
            
        except Exception as e:
            logger.error(f"Error processing game: {e}", exc_info=True)

    async def _capture_game_video(self, table) -> Optional[List[np.ndarray]]:
        """Capture video during game play"""
        try:
            cap = cv2.VideoCapture(0)
            frames = []
            
            while len(frames) < self.video_encoder.video_config.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add game state visualization
                self._draw_game_state(frame, table)
                frames.append(frame)
                
                cv2.imshow('Poker Game', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            return frames
            
        except Exception as e:
            logger.error(f"Video capture error: {e}")
            return None
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()

    def _draw_game_state(self, frame: np.ndarray, table) -> None:
        """Draw game state information on video frame"""
        # Draw player information
        y_offset = 30
        for player in table.players:
            text = f"{player.name}: {player.chips} chips"
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        # Draw community cards
        if table.community_cards:
            cards_text = f"Community: {', '.join(str(c) for c in table.community_cards)}"
            cv2.putText(frame, cards_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    async def _run_game(self, table) -> None:
        """Run poker game rounds"""
        # Deal initial cards
        for player in table.players:
            logger.info(f"\n{player.name} ({player.historical_persona})")
            logger.info(f"Starting chips: {player.chips}")
            logger.info(f"Cards: {', '.join(str(card) for card in player.cards)}")
            
            for card in player.cards:
                self.audio.add_card_sound(card)
            await asyncio.sleep(0.5)
        
        # Play through rounds
        rounds = [("Pre-flop", 0), ("Flop", 3), ("Turn", 1), ("River", 1)]
        min_bet = 100
        
        for round_name, num_cards in rounds:
            logger.info(f"\n--- {round_name} ---")
            
            if num_cards > 0:
                if round_name == "Flop": table.deal_flop()
                elif round_name == "Turn": table.deal_turn()
                else: table.deal_river()
                
                logger.info(f"Community cards: {', '.join(str(card) for card in table.community_cards)}")
            
            await self._play_betting_round(table, round_name, min_bet)
            min_bet *= 2
        
        # Handle showdown
        await self._handle_showdown(table)

    async def _play_betting_round(self, table, round_name: str, min_bet: int) -> None:
        """Play a single betting round"""
        active_players = [p for p in table.players if p.cards]
        for player in active_players:
            # Simulate AI decision
            action = self._simulate_action(player, min_bet)
            table.process_action(player, action[0], action[1])
            
            logger.info(f"{player.name} {action[0].value}s" + 
                       (f" {action[1]} chips" if action[1] else ""))
            
            if action[1]:
                self.audio.add_chip_sound(action[1])
            
            await asyncio.sleep(0.5)

    def _simulate_action(self, player, min_bet: int) -> tuple[PlayerAction, Optional[int]]:
        """Generate AI player action"""
        import random
        choice = random.random()
        
        if choice < 0.1:
            return PlayerAction.FOLD, None
        elif choice < 0.3:
            return PlayerAction.RAISE, min_bet * random.randint(2, 4)
        else:
            return PlayerAction.CALL, min_bet

    async def _handle_showdown(self, table) -> None:
        """Process game showdown"""
        logger.info("\n=== SHOWDOWN ===")
        winner, hand_rank, winning_cards = table.determine_winner()
        
        if winner:
            logger.info(f"\n🏆 Winner: {winner.name} ({winner.historical_persona})")
            logger.info(f"Winning hand: {hand_rank.value}")
            logger.info(f"Cards: {', '.join(str(card) for card in winning_cards)}")
            
            if table.oracle_commentary:
                logger.info("\nOracle Commentary:")
                for comment in table.oracle_commentary[-3:]:
                    logger.info(f"- {comment.prophecy}")
                    logger.info(f"  {comment.historical_parallel}")

async def main():
    try:
        # Initialize game manager
        manager = EnhancedGameManager()
        
        # Set up players
        players_data = [
            {"name": "Julius Caesar", "persona": "Roman Emperor", "chips": 2000},
            {"name": "Cleopatra", "persona": "Egyptian Queen", "chips": 2000},
            {"name": "Alexander", "persona": "Macedonian Conqueror", "chips": 2000}
        ]
        
        # Run game with video processing
        await manager.process_game(players_data)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())