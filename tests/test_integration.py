"""
Integration test for using card_ordering_rules.py with consolidated sort_logic.py

Tests that the card ordering rules work correctly with the sorting algorithm.
"""

import unittest
import copy
import card_ordering_rules as cor
from sort_logic import optimal_sort

class TestIntegration(unittest.TestCase):
    def test_sort_hearts_with_bottom(self):
        """Test sorting a set of hearts cards with 2 piles and allow_bottom=True."""
        hearts_mapping = cor.get_sort_mapping('hearts')
        cards = ['4h', '3h', '2h', '6h', '5h']
        card_values = [hearts_mapping.card_to_value(card.upper()) for card in cards]
        original_card_values = card_values[:]
        result = optimal_sort(card_values, num_piles=2, allow_bottom=True)
        deck = result.history[-1]
        # Check that the result is sorted numerically according to mapping values
        assert deck == sorted(original_card_values), f"Deck not sorted: {deck} vs {sorted(original_card_values)}"
        # Convert the sorted values back to cards
        sorted_cards = [hearts_mapping.value_to_card(value) for value in deck]
        expected_cards = sorted([card.upper() for card in cards], key=hearts_mapping.card_to_value)
        assert sorted_cards == expected_cards, f"Cards not sorted: {sorted_cards} vs {expected_cards}"
    
    def test_sort_with_bridge_values(self):
        """Test sorting a set of bridge cards using the sorting algorithm."""
        # Get the bridge mapping
        bridge_mapping = cor.get_sort_mapping('bridge')
        
        # A smaller set of cards (to avoid performance issues with large state space)
        cards = ['2S', 'JH', 'TD', '6C', 'QS']
        
        # Convert to numerical values using the bridge mapping
        card_values = [bridge_mapping.card_to_value(card) for card in cards]
        
        # Make a copy to verify later
        original_card_values = copy.deepcopy(card_values)
        
        # Use consolidated algorithm (caps piles at 2 internally)
        result = optimal_sort(card_values, num_piles=2, allow_bottom=True)
        deck = result.history[-1]

        # Check that the result is sorted numerically according to mapping values
        self.assertEqual(deck, sorted(original_card_values))
        
        # Convert the sorted values back to cards
        sorted_cards = [bridge_mapping.value_to_card(value) for value in deck]
        
        # Expected order by bridge rules
        expected_cards = sorted(cards, key=bridge_mapping.card_to_value)
        
        # Verify the cards are properly sorted according to bridge rules
        self.assertEqual(sorted_cards, expected_cards)
    
    def test_single_suit_sort(self):
        """Test sorting a set of cards from a single suit (2H to 6H) with 2 piles and allow_bottom=True."""
        hearts_mapping = cor.get_sort_mapping('hearts')
        cards = ['4H', '3H', '2H', '6H', '5H']
        card_values = [hearts_mapping.card_to_value(card) for card in cards]
        original_card_values = copy.deepcopy(card_values)
        result = optimal_sort(card_values, num_piles=2, allow_bottom=True)
        deck = result.history[-1]
        # Check that the result is sorted numerically according to mapping values
        self.assertEqual(deck, sorted(original_card_values))
        # Convert the sorted values back to cards
        sorted_cards = [hearts_mapping.value_to_card(value) for value in deck]
        expected_cards = sorted(cards, key=hearts_mapping.card_to_value)
        self.assertEqual(sorted_cards, expected_cards)
    
    def test_different_sort_mappings(self):
        """Test sorting the same cards with different mappings."""
        # A smaller set of cards (to avoid performance issues with large state space)
        cards = ['2S', 'JH', 'TD', '6C', 'QS']
        
        # Sort using bridge rules
        bridge_mapping = cor.get_sort_mapping('bridge')
        bridge_values = [bridge_mapping.card_to_value(card) for card in cards]
        bridge_result = optimal_sort(bridge_values, num_piles=2, allow_bottom=True)
        bridge_sorted_deck = bridge_result.history[-1]
        
        # Sort using hearts rules (different suit ordering)
        hearts_mapping = cor.get_sort_mapping('hearts')
        hearts_values = [hearts_mapping.card_to_value(card) for card in cards]
        hearts_result = optimal_sort(hearts_values, num_piles=2, allow_bottom=True)
        hearts_sorted_deck = hearts_result.history[-1]
        
        # The sorted values should be different because of different orderings
        bridge_sorted_cards = [bridge_mapping.value_to_card(value) for value in bridge_sorted_deck]
        hearts_sorted_cards = [hearts_mapping.value_to_card(value) for value in hearts_sorted_deck]
        
        # The cards should be sorted correctly according to their respective rules
        self.assertEqual(bridge_sorted_cards, sorted(cards, key=bridge_mapping.card_to_value))
        self.assertEqual(hearts_sorted_cards, sorted(cards, key=hearts_mapping.card_to_value))
        
        # The sorted orders should be different
        self.assertNotEqual(bridge_sorted_cards, hearts_sorted_cards)

if __name__ == '__main__':
    unittest.main()
