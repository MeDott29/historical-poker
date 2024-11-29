from historical_poker import create_historical_game, PlayerAction
from audio_interface.audio_interface import AudioInterface
import random
from time import sleep

# Add audio initialization at the top level
audio = AudioInterface()

def simulate_player_action(player, table, min_bet=100):
    """Simulate an AI player's decision"""
    action, amount = None, None
    
    # Generate action
    choice = random.random()
    if choice < 0.1:
        action = PlayerAction.FOLD
        # Generate fold sound
        audio.add_action_sound("fold")
    elif choice < 0.3:
        action = PlayerAction.RAISE
        amount = min_bet * random.randint(2, 4)
        # Generate raise sound
        audio.add_action_sound("raise", amount)
    else:
        action = PlayerAction.CALL
        amount = min_bet
        # Generate call sound
        audio.add_action_sound("call", amount)
        
    # Fate point sound if used
    if player.fate_points > 0 and random.random() < 0.1:
        table.process_action(player, PlayerAction.USE_FATE)
        audio.add_action_sound("fate_point")
        print(f"{player.name} uses a fate point! ({player.fate_points-1} remaining)")
    
    return action, amount

def play_betting_round(table, round_name, min_bet=100):
    """Simulate a betting round with audio feedback"""
    print(f"\n=== {round_name} Betting Round ===")
    
    # Generate round start sound
    audio.add_round_marker(round_name)
    
    active_players = [p for p in table.players if p.cards]
    if len(active_players) <= 1:
        return
    
    for player in active_players:
        # Add player turn indicator sound
        audio.add_player_marker(player.name)
        
        action, amount = simulate_player_action(player, table, min_bet)
        table.process_action(player, action, amount)
        
        action_str = f"{player.name} {action.value}s"
        if amount:
            action_str += f" {amount} chips"
        print(action_str)
        
        # Add chip sound if betting occurred
        if amount:
            audio.add_chip_sound(amount)
        
        sleep(1)

def main():
    # Initialize audio streams
    audio.initialize_streams()
    
    players_data = [
        {"name": "Julius Caesar", "persona": "Roman Emperor", "chips": 2000},
        {"name": "Cleopatra", "persona": "Egyptian Queen", "chips": 2000},
        {"name": "Alexander", "persona": "Macedonian Conqueror", "chips": 2000}
    ]

    table = create_historical_game(players_data)
    
    # Deal initial cards with card sounds
    for player in table.players:
        print(f"\n{player.name} ({player.historical_persona})")
        print(f"Starting chips: {player.chips}")
        print(f"Cards: {', '.join(str(card) for card in player.cards)}")
        
        # Generate card dealing sounds
        for card in player.cards:
            audio.add_card_sound(card)
        sleep(1)
    
    # Play through betting rounds
    rounds = [("Pre-flop", 0), ("Flop", 3), ("Turn", 1), ("River", 1)]

    min_bet = 100
    for round_name, num_cards in rounds:
        print(f"\n--- {round_name} ---")
        
        # Deal community cards
        if num_cards > 0:
            if round_name == "Flop":
                table.deal_flop()
            elif round_name == "Turn":
                table.deal_turn()
            else:
                table.deal_river()
                
            print(f"Community cards: {', '.join(str(card) for card in table.community_cards)}")
        
        play_betting_round(table, round_name, min_bet)
        min_bet *= 2
        
        if table.oracle_commentary:
            print("\nOracle speaks:")
            print(table.oracle_commentary[-1].prophecy)
        
        sleep(1)

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

    # At the end, save the audio
    audio.save_game_audio("poker_game_audio.wav")

if __name__ == '__main__':
    main()
