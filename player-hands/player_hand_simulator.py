import unittest
from uuid import UUID
from historical_poker import PokerTable, Player, Card, HandRank

class TestPokerHandEvaluation(unittest.TestCase):
    def setUp(self):
        self.player = Player(name="Test Player", chips=1000, historical_persona="Test Persona")
        self.table = PokerTable(players=[self.player])

    def test_royal_flush_evaluation(self):
        # Arrange
        self.player.cards = [
            Card(suit="hearts", value=14),  # Ace of hearts
            Card(suit="hearts", value=13),  # King of hearts
        ]
        self.table.community_cards = [
            Card(suit="hearts", value=12),  # Queen of hearts
            Card(suit="hearts", value=11),  # Jack of hearts
            Card(suit="hearts", value=10),  # 10 of hearts
            Card(suit="clubs", value=2),    # 2 of clubs (irrelevant)
            Card(suit="diamonds", value=3),  # 3 of diamonds (irrelevant)
        ]

        # Act
        hand_rank, best_cards = self.table.evaluate_hand(self.player)

        # Assert
        self.assertEqual(hand_rank, HandRank.ROYAL_FLUSH)
        self.assertEqual(len(best_cards), 5)
        self.assertTrue(all(card.suit == "hearts" for card in best_cards))
        self.assertEqual(sorted([card.value for card in best_cards], reverse=True), 
                        [14, 13, 12, 11, 10])

if __name__ == '__main__':
    unittest.main()
from historical_poker import PokerTable, Player, Card, HandRank

def simulate_poker_hand():
    """Simulates a poker hand and evaluates the result"""
    # Create test player and table
    player = Player(name="Test Player", chips=1000, historical_persona="Test Persona")
    table = PokerTable(players=[player])

    # Deal example royal flush hand
    player.cards = [
        Card(suit="hearts", value=14),  # Ace of hearts
        Card(suit="hearts", value=13),  # King of hearts
    ]
    
    table.community_cards = [
        Card(suit="hearts", value=12),  # Queen of hearts
        Card(suit="hearts", value=11),  # Jack of hearts 
        Card(suit="hearts", value=10),  # 10 of hearts
        Card(suit="clubs", value=2),    # 2 of clubs
        Card(suit="diamonds", value=3),  # 3 of diamonds
    ]

    # Evaluate the hand
    hand_rank, best_cards = table.evaluate_hand(player)

    # Print results
    print(f"\nPlayer Cards: {', '.join(str(card) for card in player.cards)}")
    print(f"Community Cards: {', '.join(str(card) for card in table.community_cards)}")
    print(f"Hand Rank: {hand_rank.value}")
    print(f"Best Five Cards: {', '.join(str(card) for card in best_cards)}")

if __name__ == '__main__':
    simulate_poker_hand()
