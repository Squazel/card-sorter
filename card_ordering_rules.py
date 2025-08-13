"""
card_ordering_rules.py

Defines and manages card sort order mappings for various games.
Provides functions to convert between card string representations (e.g., 'AS', 'TD', '3C') and their corresponding numerical values according to a game's natural order.
"""

from typing import Dict, List, Callable, Tuple, NamedTuple

# --- Common card ordering constants ---

# Suit orderings
SUITS_SPADES_FIRST = ['s', 'h', 'd', 'c']  # Bridge standard (Spades, Hearts, Diamonds, Clubs)
SUITS_HEARTS_FIRST = ['h', 's', 'd', 'c']  # Hearts first variant
SUITS_BY_COLOR = ['s', 'c', 'h', 'd']      # By color (black then red)

# Rank orderings
VALUES_A_HIGH = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']  # Ace high (Bridge)
VALUES_K_HIGH = ['K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2', 'A']  # King high
VALUES_A_LOW = VALUES_K_HIGH  # Alias for clarity

class CardMapping(NamedTuple):
    """Represents a complete card mapping with conversion functions."""
    card_to_value: Callable[[str], int]
    value_to_card: Callable[[int], str]
    get_order: Callable[[], List[str]]
    suits: List[str]
    ranks: List[str]

def create_mapping(suits: List[str], ranks: List[str]) -> CardMapping:
    """
    Create a card mapping based on the given suit and rank order.
    Returns a CardMapping object with all necessary conversion functions.
    
    Args:
        suits: Ordered list of suits (e.g., ['S', 'H', 'D', 'C'])
        ranks: Ordered list of ranks (e.g., ['A', 'K', 'Q', 'J', 'T', '9',...,'2'])
    """
    card_to_value = {}
    value_to_card = {}
    
    # Normalize the case of suits and ranks in the mapping
    normalized_suits = [s.lower() for s in suits]
    normalized_ranks = [r.upper() for r in ranks]
    
    value = 1
    for suit in normalized_suits:
        for rank in normalized_ranks:
            card = f"{rank}{suit}"
            card_to_value[card] = value
            value_to_card[value] = card
            value += 1
    
    def card_to_value_fn(card: str) -> int:
        """Convert card string (e.g., 'AS', 'TD') to its numeric sort value."""
        # Normalize the case of input card before lookup
        if len(card) != 2:
            raise ValueError(f"Invalid card format: {card}")
        
        normalized_card = f"{card[0].upper()}{card[1].lower()}"
        if normalized_card in card_to_value:
            return card_to_value[normalized_card]
        
        # If not found, raise a more helpful error
        raise KeyError(f"Card '{card}' not found. Expected format is rank+suit (e.g., 'As', 'TD', '2h'). "
                      f"Valid ranks: {normalized_ranks}, valid suits: {normalized_suits}")
    
    def value_to_card_fn(value: int) -> str:
        """Convert numeric sort value to card string (e.g., 'AS', 'TD')."""
        return value_to_card[value].upper()
    
    def get_order_fn() -> List[str]:
        """Return the list of card strings in sort order."""
        return [value_to_card[v].upper() for v in range(1, len(value_to_card) + 1)]
    
    return CardMapping(
        card_to_value=card_to_value_fn,
        value_to_card=value_to_card_fn,
        get_order=get_order_fn,
        suits=suits,
        ranks=ranks
    )

# --- Game definitions ---
# Each game is defined by its suit and rank ordering
GAME_DEFINITIONS = {
    'bridge': {
        'suit_ordering': SUITS_SPADES_FIRST,
        'value_ordering': VALUES_A_HIGH,
        'description': 'Standard Bridge ordering (Spades, Hearts, Diamonds, Clubs; Ace high)'
    },
    'hearts': {
        'suit_ordering': SUITS_HEARTS_FIRST,
        'value_ordering': VALUES_A_HIGH,
        'description': 'Hearts card game ordering (Hearts first)'
    },
    'spades': {
        'suit_ordering': SUITS_SPADES_FIRST,
        'value_ordering': VALUES_A_HIGH,
        'description': 'Spades card game ordering'
    },
    'ace_low': {
        'suit_ordering': SUITS_SPADES_FIRST,
        'value_ordering': VALUES_A_LOW,
        'description': 'Variant with Ace as the lowest card'
    },
    'by_color': {
        'suit_ordering': SUITS_BY_COLOR,
        'value_ordering': VALUES_A_HIGH,
        'description': 'Suits grouped by color (black, red)'
    }
}

# --- Build the actual mappings ---
SORT_MAPPINGS = {
    game_name: create_mapping(
        suits=definition['suit_ordering'],
        ranks=definition['value_ordering']
    )
    for game_name, definition in GAME_DEFINITIONS.items()
}

def get_sort_mapping(name: str = 'bridge'):
    """Return the mapping functions for the given sort mapping name."""
    return SORT_MAPPINGS[name]
