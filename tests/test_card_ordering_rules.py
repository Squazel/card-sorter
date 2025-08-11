"""
Test suite for card_ordering_rules.py

Tests the card mapping functionality including conversion between card strings and values,
as well as proper ordering of cards according to different game rules.
"""

import unittest
import card_ordering_rules as cor

class TestCardOrderingRules(unittest.TestCase):
    
    def test_bridge_mapping_conversion(self):
        """Test the basic card to value and value to card conversions for bridge mapping."""
        bridge_mapping = cor.get_sort_mapping('bridge')
        
        # Test card_to_value
        self.assertEqual(bridge_mapping.card_to_value('AS'), 1)  # Ace of Spades should be 1
        self.assertEqual(bridge_mapping.card_to_value('KS'), 2)  # King of Spades should be 2
        self.assertEqual(bridge_mapping.card_to_value('QH'), 16)  # Queen of Hearts
        self.assertEqual(bridge_mapping.card_to_value('2C'), 52)  # Two of Clubs should be 52
        
        # Test value_to_card
        self.assertEqual(bridge_mapping.value_to_card(1), 'AS')  # 1 should be Ace of Spades
        self.assertEqual(bridge_mapping.value_to_card(2), 'KS')  # 2 should be King of Spades
        self.assertEqual(bridge_mapping.value_to_card(31), 'TD')  # 31 should be 10 of Diamonds
        self.assertEqual(bridge_mapping.value_to_card(52), '2C')  # 52 should be 2 of Clubs
    
    def test_bridge_ordering(self):
        """Test that the ordering of cards in bridge is correct."""
        bridge_mapping = cor.get_sort_mapping('bridge')
        order = bridge_mapping.get_order()
        
        # Check first and last cards
        self.assertEqual(order[0], 'AS')  # First card should be Ace of Spades
        self.assertEqual(order[-1], '2C')  # Last card should be 2 of Clubs
        
        # Check suit boundaries
        self.assertEqual(order[12], '2S')  # Last Spade
        self.assertEqual(order[13], 'AH')  # First Heart
        self.assertEqual(order[25], '2H')  # Last Heart
        self.assertEqual(order[26], 'AD')  # First Diamond
        self.assertEqual(order[38], '2D')  # Last Diamond
        self.assertEqual(order[39], 'AC')  # First Club
        
        # Check ordering within each suit
        spades = order[:13]
        hearts = order[13:26]
        diamonds = order[26:39]
        clubs = order[39:]
        
        expected_rank_order = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        
        # Check that the rank order is correct within each suit
        for i, rank in enumerate(expected_rank_order):
            self.assertEqual(spades[i][0], rank)
            self.assertEqual(hearts[i][0], rank)
            self.assertEqual(diamonds[i][0], rank)
            self.assertEqual(clubs[i][0], rank)
    
    def test_hearts_mapping(self):
        """Test the Hearts card game ordering."""
        hearts_mapping = cor.get_sort_mapping('hearts')
        
        # In Hearts game, Hearts suit comes first
        self.assertEqual(hearts_mapping.card_to_value('AH'), 1)  # Ace of Hearts should be 1
        self.assertEqual(hearts_mapping.card_to_value('AS'), 14)  # Ace of Spades should be 14
        
        order = hearts_mapping.get_order()
        self.assertEqual(order[0], 'AH')  # First card should be Ace of Hearts
        self.assertEqual(order[12], '2H')  # Last Heart
        self.assertEqual(order[13], 'AS')  # First Spade
    
    def test_ace_low_mapping(self):
        """Test the Ace-low variant ordering."""
        ace_low_mapping = cor.get_sort_mapping('ace_low')
        
        # In Ace-low variant, Ace is the lowest card
        self.assertEqual(ace_low_mapping.card_to_value('KS'), 1)  # King of Spades should be 1
        self.assertEqual(ace_low_mapping.card_to_value('AS'), 13)  # Ace of Spades should be 13
        
        order = ace_low_mapping.get_order()
        self.assertEqual(order[0], 'KS')  # First card should be King of Spades
        self.assertEqual(order[12], 'AS')  # Last Spade should be Ace of Spades
    
    def test_by_color_mapping(self):
        """Test the by-color ordering."""
        color_mapping = cor.get_sort_mapping('by_color')
        
        # In by-color ordering, black suits come first (S, C) then red suits (H, D)
        order = color_mapping.get_order()
        self.assertEqual(order[0], 'AS')  # First card should be Ace of Spades
        self.assertEqual(order[13], 'AC')  # First Club should be Ace of Clubs
        self.assertEqual(order[26], 'AH')  # First Heart should be Ace of Hearts
        self.assertEqual(order[39], 'AD')  # First Diamond should be Ace of Diamonds
    
    def test_invalid_card(self):
        """Test handling of invalid card strings."""
        bridge_mapping = cor.get_sort_mapping('bridge')
        
        with self.assertRaises(KeyError):
            bridge_mapping.card_to_value('XS')  # Invalid rank
        
        with self.assertRaises(KeyError):
            bridge_mapping.card_to_value('AX')  # Invalid suit
    
    def test_invalid_value(self):
        """Test handling of invalid card values."""
        bridge_mapping = cor.get_sort_mapping('bridge')
        
        with self.assertRaises(KeyError):
            bridge_mapping.value_to_card(0)  # Below range
            
        with self.assertRaises(KeyError):
            bridge_mapping.value_to_card(53)  # Above range
    
    def test_invalid_mapping_name(self):
        """Test handling of invalid mapping name."""
        with self.assertRaises(KeyError):
            cor.get_sort_mapping('invalid_game')

if __name__ == '__main__':
    unittest.main()
