
"""
sort_logic.py

Consolidated card sorting algorithm for the Card Sorter project.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import heapq

# ---------- Core types ----------
@dataclass
class Move:
    """Represents a move of a card to a specific pile location."""
    card: int
    where: str  # 'P1-T','P1-B','P2-T','P2-B',...,'PN-T','PN-B','L' (Leftover)


@dataclass
class PassPlan:
    """Represents a full plan for one pass of sorting."""
    config: Tuple[str, ...]         # e.g. ('B',) or ('T','B') with 'T' = top (reverse), 'B' = bottom (preserve)
    moves: List[Move]               # per card decisions in scan order
    next_deck: List[int]            # deck after recombining
    piles_snapshot: Dict[str, List[int]]  # recorded as they will appear when picked up into next_deck


@dataclass
class SortResult:
    """Complete result of the sorting algorithm."""
    iterations: int
    plans: List[PassPlan]
    explanations: List[str]
    history: List[List[int]]


# ---------- Helper functions ----------
def is_sorted(deck: List[int]) -> bool:
    """Check if a deck is sorted in ascending order."""
    return all(deck[i] <= deck[i+1] for i in range(len(deck)-1))


def sortedness_heuristic(deck: List[int]) -> float:
    """Return a score in [0,1] indicating how sorted the deck is (1 = sorted)."""
    n = len(deck)
    if n <= 1:
        return 1.0
    inversions = 0
    for i in range(n):
        for j in range(i+1, n):
            if deck[i] > deck[j]:
                inversions += 1
    max_inv = n * (n - 1) // 2
    return 1.0 - (inversions / max_inv if max_inv else 0.0)


def validate_input(deck: List[int]) -> None:
    """Validate the input deck to ensure it contains distinct natural numbers."""
    if any(x <= 0 or not isinstance(x, int) for x in deck):
        raise ValueError("All cards must be distinct positive integers.")
    if len(deck) != len(set(deck)):
        raise ValueError("All cards must be distinct.")

def can_place_on(card: int, last: Optional[int], orientation: str) -> bool:
    """Determine if card can be placed on a pile based on orientation."""
    if last is None:
        return True
    return (card > last) if orientation == 'B' else (card < last)


def materialize_pile_for_pickup(orientation: str, seq: List[int]) -> List[int]:
    """Convert a pile to its pickup order based on orientation."""
    return seq if orientation == 'B' else list(reversed(seq))


def generate_all_configs(num_piles: int, allow_bottom: bool) -> List[Tuple[str, ...]]:
    """
    Generate pile configurations.
    - For 1 pile: [('T',)] if not allow_bottom else [('T',), ('B',)].
    - For 2 piles: [('T','T')] if not allow_bottom else all 4 combinations.
    - For >2 piles: fall back to Cartesian product respecting allow_bottom (though max is capped to 2 elsewhere).
    """
    if num_piles <= 0:
        raise ValueError("Number of piles must be positive")

    if num_piles == 1:
        return [('T',)] if not allow_bottom else [('T',), ('B',)]

    if num_piles == 2:
        if allow_bottom:
            return [('T','T'), ('T','B'), ('B','T'), ('B','B')]
        else:
            return [('T','T')]

    # Fallback for >2 piles
    options = ['T'] if not allow_bottom else ['T', 'B']
    def build(n: int) -> List[Tuple[str, ...]]:
        if n == 1:
            return [(opt,) for opt in options]
        res = []
        for opt in options:
            for sub in build(n-1):
                res.append((opt,) + sub)
        return res
    return build(num_piles)


def one_pass(deck: List[int], config: Tuple[str, ...]) -> PassPlan:
    """
    Execute one pass with fixed pile-orientations given by config.
    
    Args:
        deck: The current deck of cards
        config: Tuple of 'T' and 'B' indicating pile orientations
        
    Returns:
        A PassPlan object with the results of this pass
    """
    assert all(c in ('T','B') for c in config)
    num_piles = len(config)
    if num_piles == 1 and config[0] == 'T':
        raise ValueError("Cannot use 1 pile with top placement only; allow_bottom must be True.")
    piles: List[List[int]] = [[] for _ in range(num_piles)]
    last_vals: List[Optional[int]] = [None] * num_piles
    moves: List[Move] = []
    leftovers: List[int] = []

    # Process each card in the deck
    for idx, x in enumerate(deck):
        # Always deal into piles in round-robin fashion
        j = idx % num_piles
        piles[j].append(x)
        last_vals[j] = x
        moves.append(Move(x, f"P{j+1}-{config[j]}"))

    # Build next deck: leftovers on top, then piles in increasing order (P1, P2, ..., PN)
    piles_for_pickup = {
        f"P{j+1}-{config[j]}": materialize_pile_for_pickup(config[j], piles[j])
        for j in range(num_piles)
    }
    next_deck = leftovers[:]
    # Add piles in increasing order (lowest number first)
    for j in range(num_piles):
        pile_key = f"P{j+1}-{config[j]}"
        next_deck.extend(piles_for_pickup[pile_key])

    return PassPlan(config=config, moves=moves, next_deck=next_deck, piles_snapshot=piles_for_pickup)


# ---------- Main sorting algorithm ----------
def optimal_sort(deck: List[int], max_piles: Optional[int] = None, allow_bottom: bool = False) -> SortResult:
    """
    Find the optimal sorting sequence for the deck using the specified constraints.
    
    Args:
        deck: List of integers to sort
        max_piles: Maximum number of piles to use (default: min(len(deck), 2))
        allow_bottom: Whether bottom placement is allowed (default: False)
        
    Returns:
        SortResult object containing plans, iterations, explanations, and history
    """
    # Validate input
    validate_input(deck)
    
    # If deck is empty or has only one card, it's already sorted
    if len(deck) <= 1:
        return SortResult(iterations=0, plans=[], explanations=[], history=[deck[:]])
    
    # If deck is already sorted, return immediately
    if is_sorted(deck):
        return SortResult(iterations=0, plans=[], explanations=[], history=[deck[:]])
    
    # Set default max_piles if not specified
    if max_piles is None:
        max_piles = min(len(deck), 2)  # Default to 2 piles or fewer if deck is smaller
    else:
        # Cap to at most 2 piles regardless of input, and never more than number of cards
        max_piles = min(max_piles, len(deck), 2)
    if max_piles == 1 and not allow_bottom:
        raise ValueError("Cannot use 1 pile unless allow_bottom is True.")
    
    # Find the minimum value and maximum value in the deck
    min_val = min(deck)
    max_val = max(deck)
    
    # Initialize algorithm variables
    start_tuple = tuple(deck)
    best_result = None
    
    # Try each valid number of piles from 1 to max_piles
    for num_piles in range(1, max_piles + 1):
        # Generate all possible configurations for this number of piles
        configs = generate_all_configs(num_piles, allow_bottom)
        
        # BFS over deck states
        queue = deque([start_tuple])
        visited = set([start_tuple])
        parent: Dict[Tuple[int, ...], Tuple[Tuple[int, ...], PassPlan]] = {}
        
        found = False
        while queue and not found:
            current_tuple = queue.popleft()
            current = list(current_tuple)
            
            # Check if current state is sorted
            if is_sorted(current):
                found = True
                # Reconstruct path
                plans: List[PassPlan] = []
                history: List[List[int]] = []
                
                # Start with the final state
                state = current_tuple
                while state != start_tuple:
                    prev, plan = parent[state]
                    plans.append(plan)
                    history.append(list(state))
                    state = prev
                
                # Reverse to get chronological order
                plans.reverse()
                history.reverse()
                history.insert(0, deck[:])  # Add starting deck
                
                # Create explanations
                explanations = []
                for i, plan in enumerate(plans):
                    config_desc = ", ".join([f"Pile {j+1}: {'top' if c == 'T' else 'bottom'}" 
                                           for j, c in enumerate(plan.config)])
                    explanations.append(f"Pass {i+1}: {num_piles} piles ({config_desc})")
                
                # Update best result if this is the first solution or has fewer iterations
                iterations = len(plans)
                if best_result is None or iterations < best_result.iterations:
                    best_result = SortResult(
                        iterations=iterations,
                        plans=plans,
                        explanations=explanations,
                        history=history
                    )
                    
                # Stop the BFS for this pile count once we find a solution
                break
            
            # Try each configuration
            for config in configs:
                plan = one_pass(current, config)
                next_tuple = tuple(plan.next_deck)
                
                if next_tuple not in visited:
                    visited.add(next_tuple)
                    parent[next_tuple] = (current_tuple, plan)
                    queue.append(next_tuple)
    
    # Return the best result found
    if best_result is not None:
        return best_result
    
    # If no solution was found
    raise ValueError("Unable to sort the deck with the given constraints. This should not happen for a finite deck.")


def advanced_optimal_sort(deck: List[int], max_piles: Optional[int] = None, allow_bottom: bool = False) -> SortResult:
    """
    Wrapper around optimal_sort that handles non-sequential inputs by mapping them to a sequential range.
    This allows the algorithm to work with any set of distinct natural numbers.
    
    Args:
        deck: List of integers to sort
        max_piles: Maximum number of piles to use
        allow_bottom: Whether bottom placement is allowed
        
    Returns:
        SortResult object containing plans, iterations, explanations, and history
    """
    # Validate input
    validate_input(deck)
    
    # If deck is already sorted, return immediately
    if is_sorted(deck):
        return SortResult(iterations=0, plans=[], explanations=[], history=[deck[:]])
    
    # Create a mapping from original values to sequential values
    original_values = sorted(set(deck))
    value_map = {val: i + 1 for i, val in enumerate(original_values)}
    reverse_map = {i + 1: val for i, val in enumerate(original_values)}
    
    # Map the deck to sequential values
    mapped_deck = [value_map[val] for val in deck]
    
    # Run the sorting algorithm on the mapped deck
    result = optimal_sort(mapped_deck, max_piles, allow_bottom)
    
    # Map the results back to original values
    original_history = []
    for state in result.history:
        original_history.append([reverse_map[val] for val in state])
    
    # Update the plans with original values
    for plan in result.plans:
        # Update the moves
        for move in plan.moves:
            move.card = reverse_map[move.card]
        
        # Update the next deck
        plan.next_deck = [reverse_map[val] for val in plan.next_deck]
        
        # Update the piles snapshot
        new_snapshot = {}
        for key, values in plan.piles_snapshot.items():
            new_snapshot[key] = [reverse_map[val] for val in values]
        plan.piles_snapshot = new_snapshot
    
    # Create a new result with original values
    return SortResult(
        iterations=result.iterations,
        plans=result.plans,
        explanations=result.explanations,
        history=original_history
    )


# ---------- User-friendly output functions ----------
def format_human_readable_plan(result: SortResult) -> List[str]:
    """
    Convert a SortResult into human-readable instructions.
    
    Args:
        result: The SortResult from optimal_sort
        
    Returns:
        List of strings with human-readable instructions
    """
    if result.iterations == 0:
        return ["Cards are already in order. No sorting needed."]
    
    instructions = [f"Starting with cards: {result.history[0]}"]
    
    for i, plan in enumerate(result.plans):
        instructions.append(f"\nPass {i+1}:")
        
        # Describe the configuration
        config_desc = ", ".join([f"Pile {j+1}: {'top' if c == 'T' else 'bottom'}" 
                                for j, c in enumerate(plan.config)])
        instructions.append(f"Configuration: {config_desc}")
        
        # Describe each move
        instructions.append("Card placements:")
        for move in plan.moves:
            if move.where == "L":
                instructions.append(f"  Place card {move.card} in leftover pile")
            else:
                pile_num, placement = move.where.split("-")
                place_desc = "on top" if placement == "T" else "at bottom"
                instructions.append(f"  Place card {move.card} {place_desc} of {pile_num}")
        
        # Describe pickup order
        instructions.append("Pickup order:")
        if plan.moves:
            leftover_cards = [move.card for move in plan.moves if move.where == "L"]
            if leftover_cards:
                instructions.append(f"  1. Leftover pile: {leftover_cards}")
            
            # List piles in reverse order (higher number first)
            for j in range(len(plan.config) - 1, -1, -1):
                pile_key = f"P{j+1}-{plan.config[j]}"
                if pile_key in plan.piles_snapshot:
                    instructions.append(f"  {len(plan.config) - j + 1 if leftover_cards else len(plan.config) - j}. "
                                      f"Pile {j+1}: {plan.piles_snapshot[pile_key]}")
        
        # Show resulting deck
        instructions.append(f"Resulting deck: {plan.next_deck}")
        
        # Check if sorted
        if i == len(result.plans) - 1:
            instructions.append("\nCards are now sorted!")
    
    return instructions


def print_sort_solution(deck: List[int], max_piles: Optional[int] = None, allow_bottom: bool = False, verbose: bool = False) -> None:
    """
    Print a complete solution for sorting the given deck.
    
    Args:
        deck: List of integers to sort
        max_piles: Maximum number of piles to use
        allow_bottom: Whether bottom placement is allowed
        verbose: Whether to print verbose logs
    """
    try:
        # Run sorting algorithm
        if verbose:
            print(f"[VERBOSE] Starting sort: deck={deck}, max_piles={max_piles}, allow_bottom={allow_bottom}")
        result = advanced_optimal_sort(deck, max_piles, allow_bottom)
        if verbose:
            print(f"[VERBOSE] SortResult: iterations={result.iterations}, history={result.history}")
        # Generate and print human-readable instructions
        instructions = format_human_readable_plan(result)
        print("\n".join(instructions))
        print(f"\nSorted in {result.iterations} passes.")
    except ValueError as e:
        print(f"Error: {e}")


# ---------- Convenience API for CLI/tests ----------
def sort_cards(deck: List[int], max_piles: int = 2, allow_bottom: bool = True, time_limit: Optional[float] = None, max_iterations: Optional[int] = None, verbose: bool = False) -> SortResult:
    """Convenience wrapper that enforces a maximum of 2 piles and validates output.
    
    Args:
        deck: List of integers to sort
        max_piles: Requested maximum piles (will be capped to 2)
        allow_bottom: Whether bottom placement is allowed (default True)
        time_limit: Ignored (for compatibility with previous API)
        max_iterations: Ignored (for compatibility with previous API)
        verbose: Whether to print verbose logs
    """
    capped_max_piles = min(max_piles, 2)
    if verbose:
        print(f"[VERBOSE] sort_cards called: deck={deck}, max_piles={max_piles}, allow_bottom={allow_bottom}")
    result = advanced_optimal_sort(deck, max_piles=capped_max_piles, allow_bottom=allow_bottom)
    if verbose:
        print(f"[VERBOSE] SortResult: iterations={result.iterations}, history={result.history}")
    expected_sorted = sorted(deck)
    if result.history and result.history[-1] != expected_sorted:
        return SortResult(
            iterations=0,
            plans=[],
            explanations=["Solution produced but final state is not sorted; please adjust parameters."],
            history=[deck] + (result.history[-1:] if result.history else [])
        )
    return result


# ---------- Example usage ----------
if __name__ == "__main__":
    # Example inputs
    example_deck = [7, 2, 10, 4, 9, 1, 5, 8, 3, 6]
    
    print("=" * 60)
    print("Example 1: Default settings (max 2 piles, top placement only)")
    print("=" * 60)
    print_sort_solution(example_deck)
    
    print("\n" + "=" * 60)
    print("Example 2: 3 piles, top placement only")
    print("=" * 60)
    print_sort_solution(example_deck, max_piles=3, allow_bottom=False)
    
    print("\n" + "=" * 60)
    print("Example 3: 2 piles, allow bottom placement")
    print("=" * 60)
    print_sort_solution(example_deck, max_piles=2, allow_bottom=True)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode")
    print("=" * 60)
    
    try:
        # Get input
        input_str = input("Enter space-separated numbers to sort: ")
        numbers = [int(x) for x in input_str.strip().split()]
        
        # Get sorting parameters
        max_piles_input = input("Maximum number of piles (default=2): ").strip()
        max_piles = int(max_piles_input) if max_piles_input else 2
        
        bottom_input = input("Allow placement at bottom of piles? (y/n, default=n): ").strip().lower()
        allow_bottom = bottom_input in ('y', 'yes')
        
        print("\nSorting Solution:")
        print_sort_solution(numbers, max_piles, allow_bottom)
        
    except ValueError as e:
        print(f"Error: {e}")
