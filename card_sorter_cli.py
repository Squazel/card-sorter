"""
card_sorter_cli.py

Command-line interface for the Card Sorter program.
Allows users to input a list of cards and select a card ordering scheme.
"""

import sys
from typing import List, Dict, Any, Tuple
from card_ordering_rules import get_sort_mapping, GAME_DEFINITIONS
from sort_logic import advanced_optimal_sort, print_sort_solution, sort_cards


def get_game_selection() -> str:
    """
    Display a simple console menu for the user to select a game ordering.
    
    Returns:
        The selected game name
    """
    # Extract game options from GAME_DEFINITIONS
    game_options = list(GAME_DEFINITIONS.keys())
    
    print("\nSelect Card Ordering Scheme")
    print("--------------------------")
    print("Enter the number of your choice:")
    
    # Display options
    for i, game in enumerate(game_options):
        description = GAME_DEFINITIONS[game].get('description', game)
        print(f"{i+1}. {game}: {description}")
    
    # Get user selection
    while True:
        try:
            choice = input("\nEnter selection (1-{}): ".format(len(game_options)))
            selection = int(choice) - 1
            
            if 0 <= selection < len(game_options):
                return game_options[selection]
            else:
                print(f"Please enter a number between 1 and {len(game_options)}")
        except ValueError:
            print("Please enter a valid number")


def parse_cards(card_string: str) -> List[str]:
    """
    Parse a comma-separated string of cards.
    
    Args:
        card_string: A string with cards separated by commas (e.g., "2h,3s,4d,Th,...")
        
    Returns:
        A list of card strings in standardized format (uppercase rank, lowercase suit)
    """
    cards = [card.strip() for card in card_string.split(',')]
    
    # Validate each card
    valid_cards = []
    for card in cards:
        if len(card) != 2:
            print(f"Error: Invalid card format '{card}'. Each card should be 2 characters.")
            continue
        
        # Normalize case: uppercase rank, lowercase suit
        rank, suit = card[0].upper(), card[1].lower()
        
        if rank not in "A23456789TJQK":
            print(f"Error: Invalid rank '{rank}' in card '{card}'.")
            continue
        
        if suit not in "shdc":
            print(f"Error: Invalid suit '{suit}' in card '{card}'.")
            continue
        
        valid_cards.append(f"{rank}{suit}")
    
    if not valid_cards:
        print("No valid cards found.")
        return []
    
    return valid_cards


def display_sorted_cards(cards: List[str], game: str):
    """
    Display the sorted cards based on the selected game rules.
    Also shows their numerical representation according to the game's ordering.
    
    Args:
        cards: List of card strings
        game: The name of the game ordering to use
    """
    # Get the mapping for the selected game
    mapping = get_sort_mapping(game)
    
    try:
        # Convert cards to values and create pairs
        card_values = [(card, mapping.card_to_value(card)) for card in cards]
        
        # Sort by value
        sorted_card_values = sorted(card_values, key=lambda x: x[1])
        sorted_cards = [card for card, _ in sorted_card_values]
        
        print(f"\nCards sorted according to '{game}' rules:")
        
        # Display with numerical values
        print("\nCard numerical representations:")
        print("Card : Value")
        print("-------------")
        for card, value in sorted_card_values:
            print(f"{card} : {value}")
            
        print("\nFull order list:")
        print(", ".join(sorted_cards))
        
        # Show cards by suit
        suits = mapping.suits
        
        print("\nOrganized by suit:")
        for suit in suits:
            # Match cards by lowercase suit to ensure case insensitivity
            suit_cards_values = [(c, v) for c, v in sorted_card_values if c[1].lower() == suit.lower()]
            if suit_cards_values:
                suit_cards = [c for c, _ in suit_cards_values]
                print(f"{suit.upper()}: {', '.join(suit_cards)}")
                
        # Return the card values for use in next steps
        return sorted_card_values
        
    except KeyError as e:
        print(f"\nError: Invalid card format found: {e}")
        print("Please make sure all cards follow the expected format for the selected game.")
        print("Debug info:")
        print(f"Cards: {cards}")
        print(f"Sample valid card format: {mapping.value_to_card(1)}")
        return []


def get_max_piles():
    """
    Prompt the user for the maximum number of piles.
    
    Returns:
        int: Maximum number of piles (1-5)
    """
    while True:
        try:
            max_piles_input = input("Maximum number of piles to deal (1-5) [default: 2]: ").strip()
            
            # Use default if empty
            if not max_piles_input:
                return 2
            
            max_piles = int(max_piles_input)
            if 1 <= max_piles <= 5:
                return max_piles
            else:
                print("Please enter a number between 1 and 5.")
        except ValueError:
            print("Please enter a valid number.")


def get_bottom_placement():
    """
    Prompt the user for whether placement at the bottom of piles is allowed.
    
    Returns:
        bool: True if placement at bottom is allowed, False otherwise
    """
    while True:
        bottom_placement = input("Allow placement at bottom of pile? (yes/no) [default: no]: ").strip().lower()
        
        # Use default if empty
        if not bottom_placement:
            return False
            
        if bottom_placement in ('y', 'yes'):
            return True
        elif bottom_placement in ('n', 'no'):
            return False
        else:
            print("Please enter 'yes' or 'no'.")


def run_sorting_algorithm(cards_with_values: List[Tuple[str, int]], piles: int, allow_bottom: bool) -> Tuple[int, List[str]]:
    """
    Run the sorting algorithm with the given parameters.
    
    Args:
        cards_with_values: List of (card, value) tuples
        piles: Number of piles to use
        allow_bottom: Whether to allow placement at bottom of piles
    
    Returns:
        Tuple of (iterations, explanations)
    """
    # Extract just the values for sorting
    values = [value for _, value in cards_with_values]
    # Create expected sorted values for validation
    expected_sorted = sorted(values)
    
    try:
        # Use the consolidated solver which caps piles at 2 and validates output
        result = sort_cards(values, max_piles=piles, allow_bottom=allow_bottom)
        
        # Validate the result
        if result.history and result.history[-1] == expected_sorted:
            return result.iterations, result.explanations
        else:
            # Fall back to the original advanced_optimal_sort just in case
            result = advanced_optimal_sort(values, max_piles=piles, allow_bottom=allow_bottom)
            if result.history and result.history[-1] == expected_sorted:
                return result.iterations, result.explanations
            else:
                raise ValueError("Failed to produce a correctly sorted result")
        
    except Exception as e:
        print(f"Error during sorting: {e}")
        return -1, [f"Error: {e}"]


def main():
    """
    Main entry point for the card sorter CLI.
    """
    print("Welcome to Card Sorter!")
    print("------------------------")
    
    # Get card input from user
    card_input = input("Enter cards as comma-separated values (e.g., 2h,3s,4d,Th,...): ")
    cards = parse_cards(card_input)
    
    if not cards:
        print("No valid cards provided. Exiting.")
        return
    
    # Get game selection using simple console menu
    game = get_game_selection()
    
    # Display sorted cards and get the sorted card values
    card_values = display_sorted_cards(cards, game)
    
    if not card_values:
        return
    
    # Get sorting parameters
    max_piles = get_max_piles()
    bottom_placement_allowed = get_bottom_placement()
    
    print(f"\nSorting Configuration:")
    print(f"- Game: {game}")
    print(f"- Maximum piles: {max_piles}")
    print(f"- Allow placement at bottom: {'Yes' if bottom_placement_allowed else 'No'}")
    
    print("\nRunning sorting analysis for different configurations...")
    print("-" * 60)
    print(f"| {'Piles':<5} | {'Bottom':<8} | {'Iterations':<10} |")
    print("-" * 60)
    
    # Loop through different configurations
    best_config = None
    min_iterations = float('inf')
    best_explanations = []
    
    for piles in range(1, max_piles + 1):
        # Determine which bottom placement options to try
        bottom_options = [True, False] if bottom_placement_allowed else [False]
        
        for allow_bottom in bottom_options:
            # Run the sorting algorithm
            iterations, explanations = run_sorting_algorithm(card_values, piles, allow_bottom)
            
            if iterations > 0 and iterations < min_iterations:
                min_iterations = iterations
                best_config = (piles, allow_bottom)
                best_explanations = explanations
                
            bottom_str = "Yes" if allow_bottom else "No"
            iter_str = str(iterations) if iterations >= 0 else "Error"
            print(f"| {piles:<5} | {bottom_str:<8} | {iter_str:<10} |")
    
    print("-" * 60)
    
    # Show the best configuration
    if best_config:
        piles, allow_bottom = best_config
        print(f"\nBest configuration: {piles} pile(s), bottom placement {'allowed' if allow_bottom else 'not allowed'}")
        print(f"Minimum iterations required: {min_iterations}")
        print("\nSorting steps:")
        for step in best_explanations:
            print(f"- {step}")
    else:
        print("\nNo valid sorting configuration found.")
    
    print("\nNote: 'T' means cards are placed on top (reverse order when picked up)")
    print("      'B' means cards are placed on bottom (preserve order when picked up)")


if __name__ == "__main__":
    main()
