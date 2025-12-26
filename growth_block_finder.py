from typing import List

def find_biggest_block(cards: List[int]) -> List[int]:
    if not cards:
        return []
    
    # map cards to a sequential 1-based index
    card_to_index = {card: idx + 1 for idx, card in enumerate(sorted(cards))}
    # get indices for actual card order
    indexed_cards = [card_to_index[card] for card in cards]
    # get best sequence based on indices
    best_sequence = _find_biggest_sequential_block(indexed_cards)
    # convert back to original card values
    selected_cards = [card for card, index in card_to_index.items() if index in best_sequence]
    # return just the selected cards in the order they appeared in the original list
    return [card for card in cards if card in selected_cards]
    

def _find_biggest_sequential_block(cards: List[int]) -> List[int]:
    """
    Find the largest "growth block" in the list of cards.
    A growth block is defined as a group of consecutive numbers within the list,
    allowing for increments or decrements of 1 beyond the previously-selected number range.
    """
    if not cards:
        return []  # Indicate no cards available
    
    best_length = 0
    best_cards = []
    
    # iterate through cards to choose the initial card
    for i in range(1, len(cards)):
        starting_card = cards[i]
        min = starting_card
        max = starting_card
        length = 1
        selected_cards = [starting_card]

        for j in range(i + 1, len(cards)):
            next_card = cards[j]
            if next_card == min - 1:
                min = next_card
                length += 1
                selected_cards.append(next_card)
            elif next_card == max + 1:
                max = next_card
                length += 1
                selected_cards.append(next_card)
    
        if length > best_length:
            best_length = length
            best_cards = selected_cards
    
    return best_cards