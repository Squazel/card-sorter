"""
input_helpers.py

Helper functions and classes for parsing and validating user input for the Card Sorter program.
"""
from typing import List, Tuple

class CardInputHelper:
    @staticmethod
    def validate_card_list(cards: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate a list of card strings for format, rank, suit, and duplicates.
        Args:
            cards: List of card strings (e.g., ["2h", "3s", "4d", "Th", ...])
        Returns:
            Tuple of (valid_cards, errors):
            - valid_cards: list of valid card strings in standardized format (uppercase rank, lowercase suit)
            - errors: list of error messages for invalid cards or duplicates
        """
        seen = set()
        valid_cards = []
        errors = []
        for card in cards:
            if len(card) != 2:
                errors.append(f"Invalid card format '{card}'. Each card should be 2 characters.")
                continue
            rank, suit = card[0].upper(), card[1].lower()
            normalized = f"{rank}{suit}"
            if normalized in seen:
                errors.append(f"Duplicate card detected: '{normalized}'")
                continue
            if rank not in CardInputHelper.VALID_RANKS:
                errors.append(f"Invalid rank '{rank}' in card '{card}'.")
                continue
            if suit not in CardInputHelper.VALID_SUITS:
                errors.append(f"Invalid suit '{suit}' in card '{card}'.")
                continue
            seen.add(normalized)
            valid_cards.append(normalized)
        return valid_cards, errors
    VALID_RANKS = "A23456789TJQK"
    VALID_SUITS = "shdc"

    @staticmethod
    def parse_cards(card_string: str) -> Tuple[List[str], List[str]]:
        """
        Parse a comma-separated string of cards and validate them.
        Args:
            card_string: A string with cards separated by commas (e.g., "2h,3s,4d,Th,...")
        Returns:
            Tuple of (valid_cards, errors):
            - valid_cards: list of card strings in standardized format (uppercase rank, lowercase suit)
            - errors: list of error messages for invalid cards
        """
        cards = [card.strip() for card in card_string.split(',')]
        valid_cards = []
        errors = []
        for card in cards:
            if len(card) != 2:
                errors.append(f"Invalid card format '{card}'. Each card should be 2 characters.")
                continue
            rank, suit = card[0].upper(), card[1].lower()
            if rank not in CardInputHelper.VALID_RANKS:
                errors.append(f"Invalid rank '{rank}' in card '{card}'.")
                continue
            if suit not in CardInputHelper.VALID_SUITS:
                errors.append(f"Invalid suit '{suit}' in card '{card}'.")
                continue
            valid_cards.append(f"{rank}{suit}")
        return valid_cards, errors

    @staticmethod
    def validate_max_piles(max_piles_input: str, default: int = 2) -> Tuple[int, str]:
        """
        Validate the maximum number of piles input.
        Returns:
            (max_piles, error_message)
        """
        if not max_piles_input:
            return default, ""
        try:
            max_piles = int(max_piles_input)
            if 1 <= max_piles <= 5:
                return max_piles, ""
            else:
                return default, "Please enter a number between 1 and 5."
        except ValueError:
            return default, "Please enter a valid number."

    @staticmethod
    def validate_bottom_placement(bottom_placement_input: str, default: bool = False) -> Tuple[bool, str]:
        """
        Validate the bottom placement input.
        Returns:
            (allow_bottom, error_message)
        """
        val = bottom_placement_input.strip().lower()
        if not val:
            return default, ""
        if val in ('y', 'yes'):
            return True, ""
        elif val in ('n', 'no'):
            return False, ""
        else:
            return default, "Please enter 'yes' or 'no'."
