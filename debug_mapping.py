#!/usr/bin/env python3
import card_ordering_rules as cor

bridge = cor.get_sort_mapping('bridge')
print(f'Bridge suits: {bridge.suits}')
print(f'Bridge ranks: {bridge.ranks}')
print(f'QH value: {bridge.card_to_value("QH")}')
print(f'Value 15 card: {bridge.value_to_card(15)}')

# Print the full mapping for hearts
print("\nHearts mapping:")
for rank in bridge.ranks:
    card = f"{rank}H"
    value = bridge.card_to_value(card)
    print(f"{card}: {value}")
