#!/usr/bin/env python3
import card_ordering_rules as cor

bridge = cor.get_sort_mapping('bridge')
print(f'Bridge suits: {bridge.suits}')
print(f'Bridge ranks: {bridge.ranks}')

# Print full mapping for hearts and diamonds
print("\nHearts mapping:")
for rank in bridge.ranks:
    card = f"{rank}H"
    value = bridge.card_to_value(card)
    print(f"{card}: {value}")

print("\nDiamonds mapping:")
for rank in bridge.ranks:
    card = f"{rank}D"
    value = bridge.card_to_value(card)
    print(f"{card}: {value}")

# Print the value 27 card
print(f"\nValue 27 card: {bridge.value_to_card(27)}")
print(f"TD card value: {bridge.card_to_value('TD')}")
