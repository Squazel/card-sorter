"""
card_sorter_cli.py

Command-line interface for the Card Sorter program.
Allows users to input a list of cards and select a card ordering scheme.
"""

import sys
from typing import List, Dict, Any, Tuple
from input_helpers import CardInputHelper
from card_ordering_rules import get_sort_mapping, GAME_DEFINITIONS
from sort_logic import print_sort_solution, sort_cards, format_human_readable_plan


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
    print("Enter the number of your choice (or press Enter for option 1):")
    
    # Display options
    for i, game in enumerate(game_options):
        description = GAME_DEFINITIONS[game].get('description', game)
        print(f"{i+1}. {game}: {description}")
    
    # Get user selection
    while True:
        try:
            choice = input("\nEnter selection (1-{}): ".format(len(game_options))).strip()
            # Use default if empty
            if not choice:
                selection = 0
            else:
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
    # Parse input string into card list
    parsed_cards, parse_errors = CardInputHelper.parse_cards(card_string)
    # Validate the parsed card list for duplicates and correctness
    valid_cards, errors = CardInputHelper.validate_card_list(parsed_cards)
    # Combine errors from parsing and validation
    all_errors = parse_errors + errors
    for error in all_errors:
        print(f"Error: {error}")
    if not valid_cards:
        print("No valid cards found.")
        return []
    return valid_cards


def display_sorted_cards(cards: List[str], mapping):
    """
    Display the sorted cards based on the provided mapping.
    Args:
        cards: List of card strings
        mapping: The mapping object to use
    """
    try:
        sorted_cards = sorted(cards, key=mapping.card_to_value)
        print(f"\nCards sorted according to mapping rules:")
        print(", ".join(sorted_cards))
        # Return list of (card, value) pairs for compatibility
        return [(card, mapping.card_to_value(card)) for card in sorted_cards]
    except KeyError as e:
        print(f"\nError: Invalid card format found: {e}")
        print("Please make sure all cards follow the expected format for the selected game.")
        return []


def get_max_piles():
    """
    Prompt the user for the maximum number of piles.
    
    Returns:
        int: Maximum number of piles (1-5)
    """
    while True:
        max_piles_input = input("Maximum number of piles to deal (1-5) [default: 2]: ").strip()
        max_piles, error = CardInputHelper.validate_max_piles(max_piles_input)
        if not error:
            return max_piles
        else:
            print(error)


def get_bottom_placement():
    """
    Prompt the user for whether placement at the bottom of piles is allowed.
    
    Returns:
        bool: True if placement at bottom is allowed, False otherwise
    """
    while True:
        bottom_placement_input = input("Allow placement at bottom of pile? (yes/no) [default: yes]: ")
        if bottom_placement_input.strip() == "":
            return True
        allow_bottom, error = CardInputHelper.validate_bottom_placement(bottom_placement_input)
        if not error:
            return allow_bottom
        else:
            print(error)


def run_sorting_algorithm(cards_with_values: List[Tuple[str, int]], piles: int, allow_bottom: bool) -> Tuple[int, List[str]]:
    """
    Run the sorting algorithm with the given parameters.
    
    Args:
        cards_with_values: List of (card, value) tuples
        piles: Number of piles to use
        allow_bottom: Whether to allow placement at bottom of piles
    
    Returns:
        Tuple of (passes, explanations)
    """
    # Extract just the values for sorting
    values = [value for _, value in cards_with_values]
    # Create expected sorted values for validation
    expected_sorted = sorted(values)
    
    try:
        # Use the consolidated solver which caps piles at 2 and validates output
        result = sort_cards(values, num_piles=piles, allow_bottom=allow_bottom)
        
        # Validate the result
        if result.history and result.history[-1] == expected_sorted:
            return result.passes, result.explanations
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
    
    # Get card input from user, re-prompting if duplicates detected
    cards = []
    while not cards:
        card_input = input("Enter cards as comma-separated values (e.g., 2h,3s,4d,Th,...): ")
        cards = parse_cards(card_input)
        
        if not cards:
            print("No valid cards provided. Please try again.\n")
    
    # Get game selection using simple console menu
    game = get_game_selection()
    mapping = get_sort_mapping(game)
    
    # Display sorted cards and get the sorted card values for validation
    sorted_card_values = display_sorted_cards(cards, mapping)

    if not sorted_card_values:
        return

    # Convert the cards to values (in original input order)
    original_card_values = [(card, mapping.card_to_value(card)) for card in cards]

    # Get sorting parameters
    max_piles = get_max_piles()
    bottom_placement_allowed = get_bottom_placement()

    print(f"\nSorting Configuration:")
    print(f"- Game: {game}")
    print(f"- Maximum piles: {max_piles}")
    print(f"- Allow placement at bottom: {'Yes' if bottom_placement_allowed else 'No'}")

    print("\nRunning sorting analysis for different configurations...")
    print("-" * 60)
    print(f"| {'Piles':<5} | {'Bottom':<8} | {'Passes':<10} |")
    print("-" * 60)

    # Loop through different configurations
    best_config = None
    min_passes = float('inf')
    best_explanations = []

    for piles in range(1, max_piles + 1):
        # Determine which bottom placement options to try
        bottom_options = [True, False] if bottom_placement_allowed else [False]

        for allow_bottom in bottom_options:
            if piles == 1 and not allow_bottom:
                continue  # Skip invalid configuration

            # Run the sorting algorithm
            num_passes, explanations = run_sorting_algorithm(original_card_values, piles, allow_bottom)

            if num_passes > 0 and num_passes < min_passes:
                min_passes = num_passes
                best_config = (piles, allow_bottom)
                best_explanations = explanations

            bottom_str = "Yes" if allow_bottom else "No"
            passes_str = str(num_passes) if num_passes >= 0 else "Error"
            print(f"| {piles:<5} | {bottom_str:<8} | {passes_str:<10} |")

    print("-" * 60)

    # Show the best configuration
    if best_config:
        piles, allow_bottom = best_config
        print(f"\nBest configuration: {piles} pile(s), bottom placement {'allowed' if allow_bottom else 'not allowed'}")
        print(f"Minimum passes required: {min_passes}")
        
        # Get detailed instructions
        values = [value for _, value in original_card_values]
        try:
            result = sort_cards(values, num_piles=piles, allow_bottom=allow_bottom)
            
            # Show detailed instructions including pickup order
            print("\n" + "="*60)
            print("DETAILED SORTING INSTRUCTIONS")
            print("="*60)
            instructions = format_human_readable_plan(result)
            for line in instructions:
                print(line)
            
            # Also show compact table view
            print("\n" + "="*60)
            print("COMPACT VIEW - Card Placements")
            print("="*60)
            print("Pass | " + " | ".join([f"Card {i+1}" for i in range(len(cards))]))
            print("-" * (7 + len(cards) * 12))
            steps = result.get_standard_steps(len(cards))
            for pass_num, row in enumerate(steps, 1):
                print(f"{pass_num:<4} | " + " | ".join(row))
        except Exception as e:
            print(f"Error displaying sorting steps: {e}")
    else:
        print("\nNo valid sorting configuration found.")

    print("\n" + "="*60)
    print("LEGEND")
    print("="*60)
    print("'T' means cards are placed on top (reverse order when picked up)")
    print("'B' means cards are placed on bottom (preserve order when picked up)")
    print("First card to empty pile shows just pile number (e.g., 'P1')")
    print("Subsequent cards show pile and placement (e.g., 'P1-T' or 'P1-B')")


if __name__ == "__main__":
    main()
