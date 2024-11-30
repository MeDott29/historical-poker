from historical_poker import create_historical_game, PlayerAction, ConversationMetrics
from audio_interface import AudioInterface
from typing import Dict, List
import random
import numpy as np
from time import sleep

class GameSimulator:
    def __init__(self):
        self.audio = AudioInterface()
        self.metrics = ConversationMetrics(
            entropy=5.95,
            coherence=-523.41,
            opposition_strength=2.28
        )
        
    def simulate_player_action(self, player, table, min_bet=100):
        action, amount = None, None
        
        # Factor in current metrics to action probability
        coherence_factor = abs(self.metrics.coherence) / 1000
        entropy_factor = self.metrics.entropy / 10
        
        # Weighted choice based on metrics
        choice = random.random() * (coherence_factor + entropy_factor)
        
        if choice < 0.1:
            action = PlayerAction.FOLD
            self.audio.add_action_sound("fold")
            
        elif choice < 0.3:
            action = PlayerAction.RAISE
            # Scale bet amount based on opposition strength
            amount = int(min_bet * (2 + self.metrics.opposition_strength))
            self.audio.add_action_sound("raise", amount)
            
        else:
            action = PlayerAction.CALL
            amount = min_bet
            self.audio.add_action_sound("call", amount)
            
        # Fate point probability affected by entropy
        if player.fate_points > 0 and random.random() < (self.metrics.entropy / 10):
            table.process_action(player, PlayerAction.USE_FATE)
            self.audio.add_action_sound("fate")
            print(f"{player.name} uses a fate point! ({player.fate_points-1} remaining)")
        
        return action, amount

    def play_betting_round(self, table, round_name, min_bet=100):
        print(f"\n=== {round_name} Betting Round ===")
        self.audio.add_round_marker(round_name)
        
        active_players = [p for p in table.players if p.cards]
        if len(active_players) <= 1:
            return
            
        for player in active_players:
            self.audio.add_player_marker(player.name)
            action, amount = self.simulate_player_action(player, table, min_bet)
            table.process_action(player, action, amount)
            
            action_str = f"{player.name} {action.value}s"
            if amount:
                action_str += f" {amount} chips"
                self.audio.add_chip_sound(amount)
            print(action_str)
            
            # Update metrics based on action
            self._update_metrics(action, amount)
            sleep(0.5)

    def _update_metrics(self, action: PlayerAction, amount: int = None):
        # Adjust metrics based on game events
        entropy_change = random.uniform(-0.1, 0.1)
        coherence_change = random.uniform(5, 15)  # Positive drift
        opposition_change = random.uniform(-0.05, 0.05)
        
        self.metrics.entropy = max(0, self.metrics.entropy + entropy_change)
        self.metrics.coherence = min(-100, self.metrics.coherence + coherence_change)
        self.metrics.opposition_strength = max(1, self.metrics.opposition_strength + opposition_change)

def main():
    simulator = GameSimulator()
    
    players_data = [
        {"name": "Julius Caesar", "persona": "Roman Emperor", "chips": 2000},
        {"name": "Cleopatra", "persona": "Egyptian Queen", "chips": 2000},
        {"name": "Alexander", "persona": "Macedonian Conqueror", "chips": 2000}
    ]

    table = create_historical_game(players_data)
    simulator.audio.initialize_streams()
    
    # Initial dealing
    for player in table.players:
        print(f"\n{player.name} ({player.historical_persona})")
        print(f"Starting chips: {player.chips}")
        print(f"Cards: {', '.join(str(card) for card in player.cards)}")
        
        for card in player.cards:
            simulator.audio.add_card_sound(card)
        sleep(0.5)
    
    # Play through betting rounds
    rounds = [("Pre-flop", 0), ("Flop", 3), ("Turn", 1), ("River", 1)]
    min_bet = 100
    
    for round_name, num_cards in rounds:
        print(f"\n--- {round_name} ---")
        
        if num_cards > 0:
            if round_name == "Flop":
                table.deal_flop()
            elif round_name == "Turn":
                table.deal_turn()
            else:
                table.deal_river()
                
            print(f"Community cards: {', '.join(str(card) for card in table.community_cards)}")
        
        simulator.play_betting_round(table, round_name, min_bet)
        min_bet *= 2
        
        if table.oracle_commentary:
            print("\nOracle speaks:")
            print(table.oracle_commentary[-1].prophecy)
            
        # Print current metrics
        print(f"\nCurrent Metrics:")
        print(f"Entropy: {simulator.metrics.entropy:.2f}")
        print(f"Coherence: {simulator.metrics.coherence:.2f}")
        print(f"Opposition: {simulator.metrics.opposition_strength:.2f}")
        
        sleep(0.5)

    # Showdown
    print("\n=== SHOWDOWN ===")
    winner, hand_rank, winning_cards = table.determine_winner()
    
    if winner:
        print(f"\n🏆 Winner: {winner.name} ({winner.historical_persona})")
        print(f"Winning hand: {hand_rank.value}")
        print(f"Cards: {', '.join(str(card) for card in winning_cards)}")
        
        print("\nFinal Oracle Commentary:")
        for comment in table.oracle_commentary[-3:]:
            print(f"- {comment.prophecy}")
            print(f"  {comment.historical_parallel}")

    simulator.audio.save_game_audio("poker_game_audio.wav")

if __name__ == '__main__':
    main()