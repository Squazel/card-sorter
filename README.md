# Card Sorter

## Overview

This project is designed to assist players of Bridge and other card games who may have difficulty holding and sorting their cards (for example, into suit and value order). In some situations, a player may require another person to help sort their hand. However, depending on the play arrangements or environment, it may not be appropriate for the assisting person to see the cards (e.g., if they will play the hand later themselves).

The Card Sorter program provides a solution by allowing the cards to be scanned (for example, using a mobile phone camera). The program then generates a series of simple moves or actions that can be followed to sort the cards into a standard order, without revealing the actual cards to the assisting person.

## Key Features
- Designed for Bridge and other card games
- Assists users who have difficulty sorting cards manually
- Maintains privacy: the sorting assistant does not see the cards
- Uses scanned images (e.g., from a mobile phone) to identify cards
- Provides step-by-step sorting instructions to achieve standard ordering

## Use Cases
- Players with limited dexterity or other physical challenges
- Environments where privacy of the hand must be maintained
- Bridge clubs, tournaments, or casual play


## Project Components

### Card Sort Order Mapping
The file `card_ordering_rules.py` defines card sort order mappings for various games and provides functions to convert between card string representations (e.g., 'AS', 'TD', '3C') and their corresponding numerical values. Multiple predefined sort orders are available including Bridge, Hearts, and variants with different suit/value orderings. The system uses a flexible approach with reusable components, making it easy to add new game-specific orderings. This allows the sorting logic to work with natural numbers instead of card representations.

### Sort Logic
The file `sort_logic.py` now contains the consolidated and improved card sorting algorithm:
- Handles arbitrary numbers of input values (not limited to 13 cards)
- Works with any set of distinct natural numbers (not necessarily sequential)
- Supports configurable number of piles (capped at 2 for reliability)
- Offers optional bottom placement capability
- Uses time limits and heuristic search for larger decks
- Generates detailed step-by-step explanations of the sorting process
- Exposes a `sort_cards` convenience API that validates the final state

All previous enhanced and v2 logic have been merged into this single file. All tests are now in `tests/test_sort_logic.py`.

### Command Line Interface
The `card_sorter_cli.py` script provides an interactive command-line interface for the card sorter:
- Allows users to input a list of cards and select a card ordering scheme
- Displays cards sorted according to the selected game's rules
- Shows numerical representations of cards based on the game's ordering
- Organizes cards by suit for easier visualization
- Provides options to configure sorting parameters (number of piles, bottom placement)

## Usage Instructions

### Using the CLI
1. Run `python card_sorter_cli.py`
2. Enter cards as comma-separated values (e.g., 2h,3s,4d,Th,...)
3. Select a card ordering scheme from the available options
4. Specify sorting parameters (number of piles, bottom placement)
5. View the sorting solution with step-by-step instructions

### Using the Sort Logic Directly
You can use the sorting algorithm directly in your code:

```python
from sort_logic import optimal_sort, print_sort_solution, sort_cards

# Example with a list of numbers
numbers = [7, 2, 10, 4, 9, 1, 5, 8, 3, 6]

# Get a sorting solution (BFS search)
result = optimal_sort(
    deck=numbers,
    max_piles=2,       # Maximum number of piles to use (capped to 2 internally)
    allow_bottom=True  # Whether to allow placement at the bottom of piles
)

# Or use the convenience API that validates the final state
validated = sort_cards(
    deck=numbers,
    max_piles=2,
    allow_bottom=True
)

# Print a human-readable solution for either result
print_sort_solution(numbers, num_piles=2, allow_bottom=True)
```

## Development

For information on development setup, testing, and contributing to this project, see the [testing README](tests/README.md).

### Continuous Integration

This project uses GitHub Actions to automatically run all unit and integration tests on every commit to any branch. The CI workflow:

- Runs on every push and pull request
- Sets up Python 3.12
- Installs all test dependencies
- Executes the complete test suite using pytest
- Generates and uploads coverage reports

You can view the test results and coverage reports in the Actions tab of the GitHub repository.

## Future Development
- Card recognition from images
- Mobile application integration
- Enhanced visualization of sorting steps
- Interactive web interface
- Support for additional card games and sorting strategies
- Optimization for special cases and large card sets

---

*This project is designed to assist players who may have difficulty manually sorting cards, while maintaining privacy by not revealing the cards to the assisting person.*
