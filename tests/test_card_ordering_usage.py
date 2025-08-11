"""
Test suite for expected usage patterns of card_ordering_rules.py

These tests demonstrate how to use the card ordering functions in practice
with examples relevant to the card sorting application.
"""

import unittest
import card_ordering_rules as cor

class TestCardOrderingUsage(unittest.TestCase):
    
    def test_sort_hand_using_mapping(self):
        """Test sorting a hand of cards using the mapping."""
        # A typical bridge hand (unsorted)
        hand = ['2H', '9S', 'AS', 'KC', '5D', 'JH', '7C', 'TD', 'QS', '3D', '8C']
        
        # Get the bridge mapping
        bridge_mapping = cor.get_sort_mapping('bridge')
        
        # Sort the hand using the mapping
        sorted_hand = sorted(hand, key=bridge_mapping.card_to_value)
        
        # Expected result when sorted by bridge rules (Spades, Hearts, Diamonds, Clubs; Ace high)
        expected = ['AS', 'QS', '9S', 'JH', '2H', 'TD', '5D', '3D', 'KC', '8C', '7C']
        
        self.assertEqual(sorted_hand, expected)
    
    def test_convert_between_different_mappings(self):
        """Test converting a card's value between different game mappings."""
        # Get mappings for different games
        bridge_mapping = cor.get_sort_mapping('bridge')
        hearts_mapping = cor.get_sort_mapping('hearts')
        ace_low_mapping = cor.get_sort_mapping('ace_low')
        
        # For example, convert Queen of Spades
        card = 'QS'
        
        # Get value in bridge mapping
        bridge_value = bridge_mapping.card_to_value(card)
        
        # Get the same card's value in hearts mapping
        hearts_value = hearts_mapping.card_to_value(card)
        
        # Get the same card's value in ace_low mapping
        ace_low_value = ace_low_mapping.card_to_value(card)
        
        # Verify the values are different due to different ordering rules
        self.assertEqual(bridge_value, 3)  # 3rd card in bridge
        self.assertEqual(hearts_value, 16)  # 16th card in hearts (hearts come first)
        self.assertEqual(ace_low_value, 2)  # 2nd card in ace_low (Ace is lowest)
    
    def test_find_highest_card_in_trick(self):
        """Test finding the highest card in a trick based on game rules."""
        # A trick in a card game (cards played in one round)
        trick = ['JS', '9H', 'KS', 'QC']
        
        # Find the highest card using bridge rules
        bridge_mapping = cor.get_sort_mapping('bridge')
        
        # Convert each card to its value
        trick_values = [(card, bridge_mapping.card_to_value(card)) for card in trick]
        
        # Find the card with the minimum value (highest rank in bridge)
        winning_card, _ = min(trick_values, key=lambda x: x[1])
        
        # In bridge ordering, KS is highest
        self.assertEqual(winning_card, 'KS')
        
        # Now try with a different ordering (ace_low)
        ace_low_mapping = cor.get_sort_mapping('ace_low')
        trick_values = [(card, ace_low_mapping.card_to_value(card)) for card in trick]
        winning_card, _ = min(trick_values, key=lambda x: x[1])
        
        # In ace_low ordering, KS is still highest because Ace is low
        self.assertEqual(winning_card, 'KS')
    
    def test_custom_mapping_creation(self):
        """Test creating a custom mapping not predefined in the module."""
        # Define a custom ordering for a new game
        custom_suits = ['D', 'H', 'C', 'S']  # Diamonds high
        custom_ranks = ['7', '8', '9', 'T', 'J', 'Q', 'K', 'A', '2', '3', '4', '5', '6']  # 7 high
        
        # Create the custom mapping
        custom_mapping = cor.create_mapping(suits=custom_suits, ranks=custom_ranks)
        
        # Test the custom mapping
        self.assertEqual(custom_mapping.card_to_value('7D'), 1)  # 7 of Diamonds is highest
        self.assertEqual(custom_mapping.card_to_value('6S'), 52)  # 6 of Spades is lowest
        
        # Check ordering
        self.assertEqual(custom_mapping.value_to_card(1), '7D')
        self.assertEqual(custom_mapping.get_order()[0], '7D')
        self.assertEqual(custom_mapping.get_order()[-1], '6S')

if __name__ == '__main__':
    unittest.main()
