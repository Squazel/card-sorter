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


## Card Sort Order Mapping
The file `card_ordering_rules.py` defines card sort order mappings for various games and provides functions to convert between card string representations (e.g., 'AS', 'TD', '3C') and their corresponding numerical values. Multiple predefined sort orders are available including Bridge, Hearts, and variants with different suit/value orderings. The system uses a flexible approach with reusable components, making it easy to add new game-specific orderings. This allows the sorting logic to work with natural numbers instead of card representations.

## Development

For information on development setup, testing, and contributing to this project, see the [testing README](tests/README.md).

## Future Development
- Card recognition from images
- Customizable sorting orders (by suit, value, etc.)
- Support for additional card games
- User-friendly interface for both the card holder and the assistant

---

*This README describes the project context and goals. Implementation details and logic will follow in future updates.*
