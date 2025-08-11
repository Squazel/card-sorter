"""
Integration test for using card_ordering_rules.py with sort_logic_chatty.py

Tests that the card ordering rules work correctly with the sorting algorithm.
"""

import unittest
import copy
import card_ordering_rules as cor
import sort_logic_chatty as slc

class TestIntegration(unittest.TestCase):
    
    def test_sort_with_bridge_values(self):
        """Test sorting a set of bridge cards using the sorting algorithm."""
        # Get the bridge mapping
        bridge_mapping = cor.get_sort_mapping('bridge')
        
        # A random set of cards (represented as strings)
        cards = ['2S', 'JH', 'TD', '6C', 'QS', '3H', '4C', '8D', '9H', '5S', 'AC', 'KD', '7S']
        
        # Convert to numerical values using the bridge mapping
        card_values = [bridge_mapping.card_to_value(card) for card in cards]
        
        # Make a copy to verify later
        original_card_values = copy.deepcopy(card_values)
        
        # Run the sorting algorithm
        print_output = False  # Don't print the sorting steps
        if print_output:
            slc.print_solution(card_values, piles=1)
            slc.print_solution(card_values, piles=2)
        
        # Use the algorithm to get the sort plans
        plans, count = slc.optimal_sort(card_values, piles=2)
        
        # Apply the plans to sort the deck
        deck = card_values
        for plan in plans:
            deck = plan.next_deck
        
        # Check that the result is sorted
        self.assertEqual(deck, sorted(original_card_values))
        
        # Convert the sorted values back to cards
        sorted_cards = [bridge_mapping.value_to_card(value) for value in deck]
        
        # Expected order by bridge rules
        expected_cards = sorted(cards, key=bridge_mapping.card_to_value)
        
        # Verify the cards are properly sorted according to bridge rules
        self.assertEqual(sorted_cards, expected_cards)
    
    def test_different_sort_mappings(self):
        """Test sorting the same cards with different mappings."""
        # The same set of cards
        cards = ['2S', 'JH', 'TD', '6C', 'QS', '3H', '4C', '8D', '9H', '5S', 'AC', 'KD', '7S']
        
        # Sort using bridge rules
        bridge_mapping = cor.get_sort_mapping('bridge')
        bridge_values = [bridge_mapping.card_to_value(card) for card in cards]
        bridge_plans, _ = slc.optimal_sort(bridge_values, piles=2)
        
        bridge_sorted_deck = bridge_values
        for plan in bridge_plans:
            bridge_sorted_deck = plan.next_deck
        
        # Sort using hearts rules (different suit ordering)
        hearts_mapping = cor.get_sort_mapping('hearts')
        hearts_values = [hearts_mapping.card_to_value(card) for card in cards]
        hearts_plans, _ = slc.optimal_sort(hearts_values, piles=2)
        
        hearts_sorted_deck = hearts_values
        for plan in hearts_plans:
            hearts_sorted_deck = plan.next_deck
        
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
