import json
import random
from typing import List, Dict
from pathlib import Path
import asyncio
from datetime import datetime
import logging

# Add logger configuration
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PlayerSimulator:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Enhanced historical eras with more detail
        self.eras = {
            "Ancient": {
                "start_year": -3000, 
                "end_year": 500,
                "cultural_traits": ["Mysticism", "Honor-bound", "Tribal"],
                "typical_stakes": (500, 2000)
            },
            "Medieval": {
                "start_year": 500, 
                "end_year": 1500,
                "cultural_traits": ["Chivalry", "Faith-driven", "Feudal"],
                "typical_stakes": (1000, 5000)
            },
            "Renaissance": {
                "start_year": 1500, 
                "end_year": 1700,
                "cultural_traits": ["Artistic", "Scientific", "Merchant"],
                "typical_stakes": (2000, 8000)
            },
            "Modern": {
                "start_year": 1700, 
                "end_year": 1900,
                "cultural_traits": ["Industrial", "Rational", "Colonial"],
                "typical_stakes": (5000, 20000)
            }
        }
        
        # Enhanced play styles with personality traits
        self.play_styles = {
            "Aggressive": {
                "raise_freq": 0.4, 
                "fold_freq": 0.1, 
                "fate_point_freq": 0.3,
                "personality": ["Bold", "Impulsive", "Dominant"],
                "risk_tolerance": 0.8
            },
            "Conservative": {
                "raise_freq": 0.2, 
                "fold_freq": 0.3, 
                "fate_point_freq": 0.1,
                "personality": ["Cautious", "Analytical", "Patient"],
                "risk_tolerance": 0.3
            },
            "Balanced": {
                "raise_freq": 0.3, 
                "fold_freq": 0.2, 
                "fate_point_freq": 0.2,
                "personality": ["Adaptable", "Strategic", "Composed"],
                "risk_tolerance": 0.5
            }
        }

    def generate_player(self, era: str) -> Dict:
        """Generate a more detailed historical player"""
        era_data = self.eras[era]
        play_style = random.choice(list(self.play_styles.keys()))
        style_data = self.play_styles[play_style]
        
        # Enhanced player attributes
        return {
            "id": str(random.getrandbits(32)),
            "name": f"Historical_Player_{random.getrandbits(16)}",
            "era": era,
            "year": random.randint(era_data["start_year"], era_data["end_year"]),
            "cultural_traits": random.sample(era_data["cultural_traits"], 2),
            "play_style": play_style,
            "personality_traits": random.sample(style_data["personality"], 2),
            "risk_tolerance": style_data["risk_tolerance"] * random.uniform(0.8, 1.2),
            "raise_frequency": style_data["raise_freq"],
            "fold_frequency": style_data["fold_freq"],
            "fate_point_frequency": style_data["fate_point_freq"],
            "starting_chips": random.randint(*era_data["typical_stakes"]),
            "expertise_level": random.uniform(0.1, 1.0),
            "historical_wins": random.randint(0, 100),
            "preferred_stakes": random.choice(["Low", "Medium", "High"]),
            "reputation_score": random.uniform(0, 100),
            "created_at": datetime.now().isoformat(),
            "special_abilities": self._generate_special_abilities(era)
        }

    def _generate_special_abilities(self, era: str) -> List[Dict]:
        """Generate era-appropriate special abilities"""
        abilities = []
        num_abilities = random.randint(1, 3)
        
        ability_pool = {
            "Ancient": ["Oracle's Wisdom", "Divine Intervention", "Tribal Alliance"],
            "Medieval": ["Royal Decree", "Knight's Honor", "Holy Blessing"],
            "Renaissance": ["Merchant's Fortune", "Artist's Insight", "Scientific Method"],
            "Modern": ["Industrial Might", "Colonial Power", "Rational Analysis"]
        }
        
        for _ in range(num_abilities):
            ability = random.choice(ability_pool[era])
            abilities.append({
                "name": ability,
                "power_level": random.uniform(1, 10),
                "cooldown": random.randint(3, 10)
            })
        
        return abilities

    async def generate_player_pool(self, players_per_era: int = 25) -> List[Dict]:
        """Generate a pool of players from different eras"""
        players = []
        for era in self.eras:
            era_players = [self.generate_player(era) for _ in range(players_per_era)]
            players.extend(era_players)
        return players

    def save_players(self, players: List[Dict], filename: str = "players.jsonl"):
        """Save players to JSONL file"""
        filepath = self.data_dir / filename
        with filepath.open('w') as f:
            for player in players:
                f.write(json.dumps(player) + '\n')

    def load_players(self, filename: str = "players.jsonl") -> List[Dict]:
        """Load players from JSONL file"""
        filepath = self.data_dir / filename
        players = []
        if filepath.exists():
            with filepath.open('r') as f:
                for line in f:
                    players.append(json.loads(line))
        return players

    async def simulate_game_history(self, players: List[Dict], num_games: int = 100):
        """Enhanced game history simulation"""
        games = []
        for game_id in range(num_games):
            game_players = players
            
            # Enhanced game data
            game = {
                "game_id": str(game_id),
                "timestamp": datetime.now().isoformat(),
                "players": [p["id"] for p in game_players],
                "winner": random.choice(game_players)["id"],
                "pot_size": random.randint(1000, 10000),
                "num_rounds": random.randint(1, 4),
                "fate_points_used": random.randint(0, len(game_players)),
                "dramatic_moments": self._generate_dramatic_moments(),
                "historical_context": self._generate_historical_context(game_players),
                "special_events": self._generate_special_events(game_players)
            }
            games.append(game)
            
        return games

    def _generate_dramatic_moments(self) -> List[Dict]:
        """Generate interesting dramatic moments during the game"""
        moments = []
        if random.random() < 0.7:  # 70% chance of dramatic moments
            num_moments = random.randint(1, 3)
            moment_types = ["Bluff Called", "Lucky Draw", "Strategic Fold", "All-In Showdown"]
            for _ in range(num_moments):
                moments.append({
                    "type": random.choice(moment_types),
                    "round": random.randint(1, 4),
                    "intensity": random.uniform(1, 10)
                })
        return moments

    def _generate_historical_context(self, players: List[Dict]) -> Dict:
        """Generate historical context for the game"""
        eras_present = set(p["era"] for p in players)
        return {
            "setting": random.choice(["Royal Court", "Merchant Guild", "Military Camp", "Sacred Temple"]),
            "era_tensions": list(eras_present),
            "cultural_impact": random.uniform(1, 10),
            "historical_significance": random.uniform(1, 10)
        }

    def save_games(self, games: List[Dict], filename: str = "game_history.jsonl"):
        """Save game history to JSONL file"""
        filepath = self.data_dir / filename
        with filepath.open('w') as f:
            for game in games:
                f.write(json.dumps(game) + '\n')

    def load_games(self, filename: str = "game_history.jsonl") -> List[Dict]:
        """Load game history from JSONL file"""
        filepath = self.data_dir / filename
        games = []
        if filepath.exists():
            with filepath.open('r') as f:
                for line in f:
                    games.append(json.loads(line))
        return games

    async def get_matchup(self, num_players: int = 4) -> List[Dict]:
        """Get a balanced group of players for a game"""
        players = self.load_players()
        if not players:
            players = await self.generate_player_pool()
            self.save_players(players)
        
        # Ensure mix of eras and play styles
        selected = []
        eras_needed = list(self.eras.keys())
        while len(selected) < num_players and eras_needed:
            era = eras_needed.pop(0)
            era_players = [p for p in players if p["era"] == era]
            if era_players:
                selected.append(random.choice(era_players))
        
        # Fill remaining slots
        while len(selected) < num_players:
            player = random.choice(players)
            if player not in selected:
                selected.append(player)
                
        return selected

async def main():
    # Initialize components
    simulator = PlayerSimulator()
    
    try:
        # Load or generate player pool
        players = simulator.load_players()
        if not players:
            players = await simulator.generate_player_pool(players_per_era=50)
            simulator.save_players(players)
        
        # Run multiple games
        num_games = 10
        for game_num in range(num_games):
            # Get balanced matchup
            matchup = await simulator.get_matchup(num_players=4)
            
            # Log game start
            logger.info(f"\nStarting Game {game_num + 1}/{num_games}")
            
            # Record game history
            await simulator.simulate_game_history(matchup, num_games=1)
            await asyncio.sleep(1)
    
    except Exception as e:
        logger.error(f"Error running games: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())