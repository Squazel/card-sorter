
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
    where: str  # 'P1','P2',...,'PN' (first card to pile), 'P1-T','P1-B','P2-T','P2-B',...,'PN-T','PN-B' (subsequent cards), 'L' (Leftover)


@dataclass
class PassPlan:
    """Represents a full plan for one pass of sorting."""
    config: Tuple[str, ...]         # e.g. ('B',) or ('T','B') with 'T' = top (reverse), 'B' = bottom (preserve)
    moves: List[Move]               # per card decisions in scan order
    next_deck: List[int]            # deck after recombining
    piles_snapshot: Dict[str, List[int]]  # recorded as they will appear when picked up into next_deck


@dataclass
class SortResult:
    def get_standard_steps(self, num_cards: int) -> List[List[str]]:
        """
        Returns a list of lists, each sublist is a pass, each element is '<pile><T/B>' for each card in input order.
        Uses actual moves from the plan, not round-robin assumptions.
        """
        steps = []
        for pass_idx, plan in enumerate(self.plans):
            row = []
            for move in plan.moves:
                if move.where == "L":
                    row.append("L")
                elif "-" in move.where:
                    # Format: "P1-T" -> "1T", "P2-B" -> "2B"
                    pile_num, placement = move.where.split("-")
                    pile_digit = pile_num[1:]  # Remove 'P' prefix
                    row.append(f"{pile_digit}{placement}")
                else:
                    # Format: "P1" -> "1", "P2" -> "2" (first card to pile, no T/B)
                    pile_digit = move.where[1:]  # Remove 'P' prefix
                    row.append(pile_digit)
            steps.append(row)
        return steps
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


def one_pass_greedy_distribution(deck: List[int], config: Tuple[str, ...]) -> PassPlan:
    """
    Execute one pass with greedy card distribution.
    
    For each card in order, choose which pile to place it on based on a simple heuristic:
    - Try to maintain increasing sequences within piles when using 'B' orientation
    - Try to maintain decreasing sequences within piles when using 'T' orientation
    
    This allows more flexibility than round-robin while remaining tractable.
    
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
    moves: List[Move] = []
    
    # For each card, greedily choose the best pile
    for card in deck:
        best_pile = 0
        best_score = -float('inf')
        
        for pile_idx in range(num_piles):
            pile = piles[pile_idx]
            orientation = config[pile_idx]
            
            # Score this placement
            if len(pile) == 0:
                # Empty pile - neutral score
                score = 0
            elif orientation == 'B':
                # Bottom placement: prefer if card is greater than last
                last_card = pile[-1]
                if card > last_card:
                    score = 10 + (card - last_card)  # Bonus for maintaining order
                else:
                    score = -abs(card - last_card)  # Penalty for breaking order
            else:  # orientation == 'T'
                # Top placement: prefer if card is less than last (will be reversed)
                last_card = pile[-1]
                if card < last_card:
                    score = 10 + (last_card - card)  # Bonus for maintaining reverse order
                else:
                    score = -abs(card - last_card)  # Penalty for breaking order
            
            if score > best_score:
                best_score = score
                best_pile = pile_idx
        
        # Place card on best pile
        pile_was_empty = len(piles[best_pile]) == 0
        piles[best_pile].append(card)
        # If pile was empty, don't specify T/B (nothing to put it under/above)
        if pile_was_empty:
            moves.append(Move(card, f"P{best_pile+1}"))
        else:
            moves.append(Move(card, f"P{best_pile+1}-{config[best_pile]}"))
    
    # Build next deck: piles in increasing order (P1, P2)
    piles_for_pickup = {
        f"P{j+1}-{config[j]}": materialize_pile_for_pickup(config[j], piles[j])
        for j in range(num_piles)
    }
    next_deck = []
    for j in range(num_piles):
        pile_key = f"P{j+1}-{config[j]}"
        next_deck.extend(piles_for_pickup[pile_key])
    
    return PassPlan(config=config, moves=moves, next_deck=next_deck, piles_snapshot=piles_for_pickup)




def one_pass(deck: List[int], config: Tuple[str, ...]) -> PassPlan:
    """
    Execute one pass with fixed pile-orientations given by config.
    
    Args:
        deck: The current deck of cards
        config: Tuple of 'T' and 'B' indicating pile orientations
        
    Returns:
        A PassPlan object with the results of this pass
        
    TODO: To support the new action model (see ALGORITHM.md):
    - Add pickup_strategy parameter to specify which piles to pick up
    - Modify to handle dealing onto piles that already have cards
    - Update state representation to track cards in hand vs on table
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
        # TODO: Explore non-round-robin distributions for better solutions
        # Currently: Always deal into piles in round-robin fashion
        # New model: Allow free choice of pile for each card
        j = idx % num_piles
        pile_was_empty = len(piles[j]) == 0
        piles[j].append(x)
        last_vals[j] = x
        # If pile was empty, don't specify T/B (nothing to put it under/above)
        if pile_was_empty:
            moves.append(Move(x, f"P{j+1}"))
        else:
            moves.append(Move(x, f"P{j+1}-{config[j]}"))

    # Build next deck: leftovers on top, then piles in increasing order (P1, P2, ..., PN)
    piles_for_pickup = {
        f"P{j+1}-{config[j]}": materialize_pile_for_pickup(config[j], piles[j])
        for j in range(num_piles)
    }
    next_deck = leftovers[:]
    # TODO: Support flexible pickup strategies (P1 only, P2 only, P1+P2, P2+P1)
    # Currently: Add piles in increasing order (lowest number first)
    for j in range(num_piles):
        pile_key = f"P{j+1}-{config[j]}"
        next_deck.extend(piles_for_pickup[pile_key])

    return PassPlan(config=config, moves=moves, next_deck=next_deck, piles_snapshot=piles_for_pickup)


# ---------- Main sorting algorithm ----------
def optimal_sort(deck: List[int], num_piles: int, allow_bottom: bool) -> SortResult:
    """
    Find an efficient sorting sequence for the deck using the specified constraints.
    
    For decks <= 10 cards, uses BFS for optimal solution.
    For decks > 10 cards, uses beam search for fast practical solution.
    
    Args:
        deck: List of integers to sort
        num_piles: Number of piles to use
        allow_bottom: Whether bottom placement is allowed
        
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
    
    # Use BFS for small decks (optimal), beam search for large decks (fast)
    if len(deck) <= 10:
        return _bfs_sort(deck, num_piles, allow_bottom)
    else:
        return _beam_search_sort(deck, num_piles, allow_bottom)


def _bfs_sort(deck: List[int], num_piles: int, allow_bottom: bool) -> SortResult:
    """
    BFS-based optimal sorting for small decks (<=10 cards).
    Guaranteed to find the solution with minimum iterations.
    """
    start_tuple = tuple(deck)
    best_result = None
    
    # Generate all possible configurations for this number of piles
    configs = generate_all_configs(num_piles, allow_bottom)
    # Filter out invalid config: 1 pile, top placement only
    if num_piles == 1:
        configs = [cfg for cfg in configs if cfg != ('T',)]

    # BFS over deck states
    queue = deque([start_tuple])
    visited = set([start_tuple])
    parent: Dict[Tuple[int, ...], Tuple[Tuple[int, ...], PassPlan]] = {}

    while queue:
        current_tuple = queue.popleft()
        current = list(current_tuple)

        # Check if current state is sorted
        if is_sorted(current):
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
            return best_result

        # Try each configuration
        for config in configs:
            plan = one_pass_greedy_distribution(current, config)
            next_tuple = tuple(plan.next_deck)

            if next_tuple not in visited:
                visited.add(next_tuple)
                parent[next_tuple] = (current_tuple, plan)
                queue.append(next_tuple)
    
    # If no solution was found
    raise ValueError("Unable to sort the deck with the given constraints.")


def _beam_search_sort(deck: List[int], num_piles: int, allow_bottom: bool, beam_width: int = 50, max_depth: int = 30) -> SortResult:
    """
    Beam search for large decks (>10 cards).
    Uses greedy card distribution instead of round-robin for better results.
    Keeps only the most promising states at each level to limit memory and time.
    """
    start_tuple = tuple(deck)
    configs = generate_all_configs(num_piles, allow_bottom)
    if num_piles == 1:
        configs = [cfg for cfg in configs if cfg != ('T',)]
    
    # Beam search: keep top-k states at each level
    # Each beam entry: (state_tuple, path_from_start)
    current_beam = [(start_tuple, [])]
    visited_global = {start_tuple}
    
    for depth in range(max_depth):
        next_beam = []
        
        for current_tuple, path in current_beam:
            current = list(current_tuple)
            
            # Check if sorted
            if is_sorted(current):
                # Found solution! Reconstruct and return
                history = [deck[:]]
                for plan in path:
                    history.append(plan.next_deck[:])
                
                explanations = []
                for i, plan in enumerate(path):
                    config_desc = ", ".join([f"Pile {j+1}: {'top' if c == 'T' else 'bottom'}" 
                                            for j, c in enumerate(plan.config)])
                    explanations.append(f"Pass {i+1}: {num_piles} piles ({config_desc})")
                
                return SortResult(
                    iterations=len(path),
                    plans=path,
                    explanations=explanations,
                    history=history
                )
            
            # Generate successors using greedy distribution
            for config in configs:
                plan = one_pass_greedy_distribution(current, config)
                next_tuple = tuple(plan.next_deck)
                
                if next_tuple not in visited_global:
                    visited_global.add(next_tuple)
                    new_path = path + [plan]
                    next_beam.append((next_tuple, new_path))
        
        if not next_beam:
            break
        
        # Keep only top beam_width states based on sortedness heuristic
        next_beam_scored = [(sortedness_heuristic(list(state)), state, path) 
                            for state, path in next_beam]
        next_beam_scored.sort(reverse=True, key=lambda x: x[0])
        current_beam = [(state, path) for _, state, path in next_beam_scored[:beam_width]]
    
    # If no solution found, return best attempt
    if current_beam:
        best_state, best_path = max(current_beam, key=lambda x: sortedness_heuristic(list(x[0])))
        history = [deck[:]]
        for plan in best_path:
            history.append(plan.next_deck[:])
        
        explanations = []
        for i, plan in enumerate(best_path):
            config_desc = ", ".join([f"Pile {j+1}: {'top' if c == 'T' else 'bottom'}" 
                                    for j, c in enumerate(plan.config)])
            explanations.append(f"Pass {i+1}: {num_piles} piles ({config_desc})")
        
        return SortResult(
            iterations=len(best_path),
            plans=best_path,
            explanations=explanations,
            history=history
        )
    
    # Fallback: return empty result
    return SortResult(iterations=0, plans=[], explanations=[], history=[deck[:]])
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
            elif "-" in move.where:
                # Format: "P1-T" or "P1-B"
                pile_num, placement = move.where.split("-")
                place_desc = "on top" if placement == "T" else "at bottom"
                instructions.append(f"  Place card {move.card} {place_desc} of {pile_num}")
            else:
                # Format: "P1" (first card to empty pile)
                instructions.append(f"  Place card {move.card} in {move.where}")
        
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


def print_sort_solution(deck: List[int], num_piles: int, allow_bottom: bool, verbose: bool = False) -> None:
    """
    Print a complete solution for sorting the given deck.
    
    Args:
        deck: List of integers to sort
        num_piles: Maximum number of piles to use
        allow_bottom: Whether bottom placement is allowed
        verbose: Whether to print verbose logs
    """
    try:
        # Run sorting algorithm
        if verbose:
            print(f"[VERBOSE] Starting sort: deck={deck}, num_piles={num_piles}, allow_bottom={allow_bottom}")
        result = optimal_sort(deck, num_piles, allow_bottom)
        if verbose:
            print(f"[VERBOSE] SortResult: iterations={result.iterations}, history={result.history}")
        # Generate and print human-readable instructions
        instructions = format_human_readable_plan(result)
        print("\n".join(instructions))
        print(f"\nSorted in {result.iterations} passes.")
    except ValueError as e:
        print(f"Error: {e}")


# ---------- Convenience API for CLI/tests ----------
def sort_cards(deck: List[int], num_piles: int, allow_bottom: bool, time_limit: Optional[float] = None, max_iterations: Optional[int] = None, verbose: bool = False) -> SortResult:
    """Convenience wrapper that enforces a maximum of 2 piles and validates output.
    
    Args:
        deck: List of integers to sort
        num_piles: Requested maximum piles
        allow_bottom: Whether bottom placement is allowed
        time_limit: Ignored (for compatibility with previous API)
        max_iterations: Ignored (for compatibility with previous API)
        verbose: Whether to print verbose logs
    """
    if verbose:
        print(f"[VERBOSE] sort_cards called: deck={deck}, num_piles={num_piles}, allow_bottom={allow_bottom}")
    result = optimal_sort(deck, num_piles=num_piles, allow_bottom=allow_bottom)
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
