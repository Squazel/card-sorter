from growth_block_finder import find_biggest_block
from card_ordering_rules import get_sort_mapping

def test_find_biggest_block():
    # Card strings: ad,8d,7d,td,jd,5d,6d,9d,qd,3d,kd,2d,4d
    card_strings = ['AD', '8D', '7D', 'TD', 'JD', '5D', '6D', '9D', 'QD', '3D', 'KD', '2D', '4D']
    
    # Use bridge mapping to convert to numbers
    mapping = get_sort_mapping('bridge')
    input_nums = [mapping.card_to_value(card) for card in card_strings]
    
    # Run the function
    result_nums = find_biggest_block(input_nums)
    
    # Convert back to card strings
    result_cards = [mapping.value_to_card(num) for num in result_nums]
    
    # Expected: 5 cards forming the biggest block: TD, JD, 9D, QD, KD
    expected = ['TD', 'JD', '9D', 'QD', 'KD']
    assert result_cards == expected