
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
    pickup_strategy: str = "ALL"    # 'ALL' (both piles), 'P1' (pile 1 only), 'P2' (pile 2 only), 'P2_FIRST' (P2 then P1)
    table_pile: Tuple[int, ...] = ()  # Cards left on table (empty tuple if none)
    table_pile_id: Optional[int] = None  # Which pile was left on table (1 or 2, None if none)


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
    passes: int  # Number of passes (dealing + pickup cycles) needed to sort
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


def lis_heuristic(deck: List[int]) -> float:
    """Return a score based on longest increasing subsequence length."""
    n = len(deck)
    if n <= 1:
        return 1.0
    
    # Compute LIS length using dynamic programming
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if deck[j] < deck[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    lis_length = max(dp)
    return lis_length / n


def lds_heuristic(deck: List[int]) -> float:
    """Return a score based on longest decreasing subsequence length."""
    n = len(deck)
    if n <= 1:
        return 1.0
    
    # Compute LDS length using dynamic programming
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if deck[j] > deck[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    lds_length = max(dp)
    return lds_length / n


def longest_monotonic_subsequence_heuristic(deck: List[int]) -> float:
    """Return a score based on longest monotonic (increasing or decreasing) subsequence."""
    if len(deck) <= 1:
        return 1.0
    
    # Take the maximum of LIS and LDS since we can use either orientation
    return max(lis_heuristic(deck), lds_heuristic(deck))


def combined_heuristic(deck: List[int]) -> float:
    """Combined heuristic using sortedness and longest monotonic subsequence."""
    # Give more weight to the monotonic subsequence since it's more relevant for pile sorting
    return 0.3 * sortedness_heuristic(deck) + 0.7 * longest_monotonic_subsequence_heuristic(deck)


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


def one_pass_greedy_distribution_with_table(
    deck: List[int], 
    config: Tuple[str, ...], 
    pickup_strategy: str = "ALL",
    table_pile: Tuple[int, ...] = (),
    table_pile_id: Optional[int] = None
) -> PassPlan:
    """
    Execute one pass with greedy card distribution, supporting pile persistence.
    
    Args:
        deck: The current deck of cards in hand
        config: Tuple of 'T' and 'B' indicating pile orientations
        pickup_strategy: How to pick up piles ('ALL', 'P1', 'P2', 'P2_FIRST')
        table_pile: Cards already on table from previous pass (in pickup order)
        table_pile_id: Which pile the table cards belong to (1 or 2, None if empty)
        
    Returns:
        A PassPlan object with the results of this pass
    """
    assert all(c in ('T','B') for c in config)
    num_piles = len(config)
    if num_piles == 1 and config[0] == 'T':
        raise ValueError("Cannot use 1 pile with top placement only; allow_bottom must be True.")
    
    piles: List[List[int]] = [[] for _ in range(num_piles)]
    
    # If there's a table pile, initialize that pile with its cards
    if table_pile_id is not None and table_pile:
        # table_pile is in pickup order, need to reverse based on orientation
        pile_idx = table_pile_id - 1
        orientation = config[pile_idx]
        # Reverse the logic: if orientation is 'B', pickup was in order, so pile is in order
        # If orientation is 'T', pickup was reversed, so we need to reverse again to get original
        if orientation == 'T':
            piles[pile_idx] = list(reversed(table_pile))
        else:
            piles[pile_idx] = list(table_pile)
    
    moves: List[Move] = []
    
    # For each card in hand, greedily choose the best pile
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
        pile_was_empty = len(piles[best_pile]) == 0 and (table_pile_id != best_pile + 1)
        piles[best_pile].append(card)
        # If pile was empty and not the table pile, don't specify T/B
        if pile_was_empty:
            moves.append(Move(card, f"P{best_pile+1}"))
        else:
            moves.append(Move(card, f"P{best_pile+1}-{config[best_pile]}"))
    
    # Build next deck based on pickup strategy
    piles_for_pickup = {
        f"P{j+1}-{config[j]}": materialize_pile_for_pickup(config[j], piles[j])
        for j in range(num_piles)
    }
    
    next_deck = []
    new_table_pile = ()
    new_table_pile_id = None
    
    if pickup_strategy == "ALL":
        # Pick up both piles: P1 first, then P2
        for j in range(num_piles):
            pile_key = f"P{j+1}-{config[j]}"
            next_deck.extend(piles_for_pickup[pile_key])
    elif pickup_strategy == "P1":
        # Pick up P1 only, leave P2 on table
        next_deck.extend(piles_for_pickup[f"P1-{config[0]}"])
        new_table_pile = tuple(piles_for_pickup[f"P2-{config[1]}"])
        new_table_pile_id = 2
    elif pickup_strategy == "P2":
        # Pick up P2 only, leave P1 on table
        next_deck.extend(piles_for_pickup[f"P2-{config[1]}"])
        new_table_pile = tuple(piles_for_pickup[f"P1-{config[0]}"])
        new_table_pile_id = 1
    elif pickup_strategy == "P2_FIRST":
        # Pick up both piles: P2 first, then P1
        for j in reversed(range(num_piles)):
            pile_key = f"P{j+1}-{config[j]}"
            next_deck.extend(piles_for_pickup[pile_key])
    
    return PassPlan(
        config=config, 
        moves=moves, 
        next_deck=next_deck, 
        piles_snapshot=piles_for_pickup,
        pickup_strategy=pickup_strategy,
        table_pile=new_table_pile,
        table_pile_id=new_table_pile_id
    )


def one_pass_greedy_distribution(deck: List[int], config: Tuple[str, ...]) -> PassPlan:
    """
    Execute one pass with greedy card distribution (backward compatible wrapper).
    
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
    # Use the enhanced function with default parameters for backward compatibility
    return one_pass_greedy_distribution_with_table(deck, config, pickup_strategy="ALL")




# ---------- Main sorting algorithm ----------
def optimal_sort(deck: List[int], num_piles: int, allow_bottom: bool, use_persistence: bool = False) -> SortResult:
    """
    Find an efficient sorting sequence for the deck using the specified constraints.
    
    For decks <= 10 cards, uses BFS for optimal solution.
    For decks > 10 cards, uses beam search or enhanced beam search with pile persistence.
    
    Args:
        deck: List of integers to sort
        num_piles: Number of piles to use
        allow_bottom: Whether bottom placement is allowed
        use_persistence: Whether to use pile persistence (allows leaving 1 pile on table)
        
    Returns:
        SortResult object containing plans, passes, explanations, and history
    """
    # Validate input
    validate_input(deck)
    
    # If deck is empty or has only one card, it's already sorted
    if len(deck) <= 1:
        return SortResult(passes=0, plans=[], explanations=[], history=[deck[:]])
    
    # If deck is already sorted, return immediately
    if is_sorted(deck):
        return SortResult(passes=0, plans=[], explanations=[], history=[deck[:]])
    
    # Use enhanced beam search with pile persistence for large decks when requested
    if use_persistence and num_piles == 2 and len(deck) > 10:
        return _beam_search_sort_with_persistence(deck, num_piles, allow_bottom, beam_width=200, max_depth=25)
    
    # Use enhanced BFS with pile persistence for small-medium decks
    if use_persistence and num_piles == 2 and len(deck) <= 10:
        return _bfs_sort_with_persistence(deck, num_piles, allow_bottom, max_depth=10)
    
    # Use BFS for small decks (optimal), beam search for large decks (fast)
    if len(deck) <= 10:
        return _bfs_sort(deck, num_piles, allow_bottom)
    else:
        return _beam_search_sort(deck, num_piles, allow_bottom)


def _bfs_sort(deck: List[int], num_piles: int, allow_bottom: bool) -> SortResult:
    """
    BFS-based optimal sorting for small decks (<=10 cards).
    Guaranteed to find the solution with minimum passes.
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

            # Update best result if this is the first solution or has fewer passes
            num_passes = len(plans)
            if best_result is None or num_passes < best_result.passes:
                best_result = SortResult(
                    passes=num_passes,
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
                    passes=len(path),
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
        
        # Keep only top beam_width states based on combined heuristic
        next_beam_scored = [(combined_heuristic(list(state)), state, path) 
                            for state, path in next_beam]
        next_beam_scored.sort(reverse=True, key=lambda x: x[0])
        current_beam = [(state, path) for _, state, path in next_beam_scored[:beam_width]]
    
    # If no solution found, return best attempt
    if current_beam:
        best_state, best_path = max(current_beam, key=lambda x: combined_heuristic(list(x[0])))
        history = [deck[:]]
        for plan in best_path:
            history.append(plan.next_deck[:])
        
        explanations = []
        for i, plan in enumerate(best_path):
            config_desc = ", ".join([f"Pile {j+1}: {'top' if c == 'T' else 'bottom'}" 
                                    for j, c in enumerate(plan.config)])
            explanations.append(f"Pass {i+1}: {num_piles} piles ({config_desc})")
        
        return SortResult(
            passes=len(best_path),
            plans=best_path,
            explanations=explanations,
            history=history
        )
    
    # Fallback: return empty result
    return SortResult(passes=0, plans=[], explanations=[], history=[deck[:]])


def _bfs_sort_with_persistence(deck: List[int], num_piles: int, allow_bottom: bool, max_depth: int = 10) -> SortResult:
    """
    Enhanced BFS with pile persistence support.
    Allows leaving one pile on the table between passes.
    State: (hand_cards, table_pile, table_pile_id)
    """
    # Generate all possible configurations
    configs = generate_all_configs(num_piles, allow_bottom)
    if num_piles == 1:
        configs = [cfg for cfg in configs if cfg != ('T',)]
    
    # Pickup strategies to explore (only for 2 piles)
    if num_piles == 2:
        pickup_strategies = ["ALL", "P1", "P2", "P2_FIRST"]
    else:
        pickup_strategies = ["ALL"]
    
    # Extended state: (hand_tuple, table_pile_tuple, table_pile_id)
    start_state = (tuple(deck), (), None)
    
    # BFS
    queue = deque([start_state])
    visited = {start_state}
    parent: Dict[Tuple, Tuple[Tuple, PassPlan]] = {}
    
    while queue:
        current_state = queue.popleft()
        hand_tuple, table_pile, table_pile_id = current_state
        hand = list(hand_tuple)
        
        # Check if everything is sorted
        # Case 1: All cards are in hand (no table pile) and hand is sorted
        if not table_pile and is_sorted(hand):
            # Reconstruct path
            plans: List[PassPlan] = []
            history: List[List[int]] = []
            
            state = current_state
            while state != start_state:
                prev_state, plan = parent[state]
                plans.append(plan)
                # For history, we track the hand state (what's in hand)
                history.append(list(state[0]))
                state = prev_state
            
            # Reverse to get chronological order
            plans.reverse()
            history.reverse()
            history.insert(0, deck[:])
            
            # Create explanations
            explanations = []
            for i, plan in enumerate(plans):
                config_desc = ", ".join([f"Pile {j+1}: {'top' if c == 'T' else 'bottom'}" 
                                        for j, c in enumerate(plan.config)])
                pickup_desc = {
                    "ALL": "pickup both piles (P1, P2)",
                    "P1": "pickup P1 only (leave P2 on table)",
                    "P2": "pickup P2 only (leave P1 on table)",
                    "P2_FIRST": "pickup both piles (P2, P1)"
                }.get(plan.pickup_strategy, plan.pickup_strategy)
                explanations.append(f"Pass {i+1}: {config_desc}, {pickup_desc}")
            
            return SortResult(
                passes=len(plans),
                plans=plans,
                explanations=explanations,
                history=history
            )
        
        # Case 2: There's a table pile - check if we can pick it up to complete sorting
        if table_pile and hand:
            combined = list(hand) + list(table_pile)
            if is_sorted(combined):
                # Reconstruct path
                plans: List[PassPlan] = []
                history: List[List[int]] = []
                
                state = current_state
                while state != start_state:
                    prev_state, plan = parent[state]
                    plans.append(plan)
                    history.append(list(state[0]))
                    state = prev_state
                
                plans.reverse()
                history.reverse()
                history.insert(0, deck[:])
                history.append(combined)
                
                # Create a final plan for picking up table pile
                final_plan = one_pass_greedy_distribution_with_table(
                    [], configs[0], "ALL", table_pile, table_pile_id
                )
                final_plan.next_deck = combined
                plans.append(final_plan)
                
                explanations = []
                for i, plan in enumerate(plans[:-1]):
                    config_desc = ", ".join([f"Pile {j+1}: {'top' if c == 'T' else 'bottom'}" 
                                            for j, c in enumerate(plan.config)])
                    pickup_desc = {
                        "ALL": "pickup both piles (P1, P2)",
                        "P1": "pickup P1 only (leave P2 on table)",
                        "P2": "pickup P2 only (leave P1 on table)",
                        "P2_FIRST": "pickup both piles (P2, P1)"
                    }.get(plan.pickup_strategy, plan.pickup_strategy)
                    explanations.append(f"Pass {i+1}: {config_desc}, {pickup_desc}")
                explanations.append(f"Pass {len(plans)}: pickup remaining table pile")
                
                return SortResult(
                    passes=len(plans),
                    plans=plans,
                    explanations=explanations,
                    history=history
                )
        
        # Don't go too deep
        # Count the depth by reconstructing path
        depth = 0
        state = current_state
        while state in parent:
            depth += 1
            state = parent[state][0]
        
        if depth >= max_depth:
            continue
        
        # Try each configuration and pickup strategy
        for config in configs:
            for pickup_strategy in pickup_strategies:
                plan = one_pass_greedy_distribution_with_table(
                    hand, config, pickup_strategy, table_pile, table_pile_id
                )
                
                # New state after this pass
                new_hand = tuple(plan.next_deck)
                new_table_pile = plan.table_pile
                new_table_pile_id = plan.table_pile_id
                new_state = (new_hand, new_table_pile, new_table_pile_id)
                
                if new_state not in visited:
                    visited.add(new_state)
                    parent[new_state] = (current_state, plan)
                    queue.append(new_state)
    
    # No solution found
    raise ValueError("Unable to sort the deck with the given constraints.")


def _beam_search_sort_with_persistence(deck: List[int], num_piles: int, allow_bottom: bool, beam_width: int = 30, max_depth: int = 10) -> SortResult:
    """
    Beam search with pile persistence support for large decks.
    Allows leaving one pile on the table between passes.
    State: (hand_cards, table_pile, table_pile_id)
    """
    # Generate all possible configurations
    configs = generate_all_configs(num_piles, allow_bottom)
    if num_piles == 1:
        configs = [cfg for cfg in configs if cfg != ('T',)]
    
    # Pickup strategies to explore (only for 2 piles)
    if num_piles == 2:
        pickup_strategies = ["ALL", "P1", "P2", "P2_FIRST"]
    else:
        pickup_strategies = ["ALL"]
    
    # Extended state: (hand_tuple, table_pile_tuple, table_pile_id)
    start_state = (tuple(deck), (), None)
    
    # Beam search: keep top-k states at each level
    # Each beam entry: (state, path_from_start)
    current_beam = [(start_state, [])]
    visited_global = {start_state}
    
    for depth in range(max_depth):
        next_beam = []
        
        for current_state, path in current_beam:
            hand_tuple, table_pile, table_pile_id = current_state
            hand = list(hand_tuple)
            
            # Check if everything is sorted
            # Case 1: All cards are in hand (no table pile) and hand is sorted
            if not table_pile and is_sorted(hand):
                # Found solution! Reconstruct and return
                history = [deck[:]]
                for plan in path:
                    history.append(plan.next_deck[:])
                
                explanations = []
                for i, plan in enumerate(path):
                    config_desc = ", ".join([f"Pile {j+1}: {'top' if c == 'T' else 'bottom'}" 
                                            for j, c in enumerate(plan.config)])
                    pickup_desc = {
                        "ALL": "pickup both piles (P1, P2)",
                        "P1": "pickup P1 only (leave P2 on table)",
                        "P2": "pickup P2 only (leave P1 on table)",
                        "P2_FIRST": "pickup both piles (P2, P1)"
                    }.get(plan.pickup_strategy, plan.pickup_strategy)
                    explanations.append(f"Pass {i+1}: {config_desc}, {pickup_desc}")
                
                return SortResult(
                    passes=len(path),
                    plans=path,
                    explanations=explanations,
                    history=history
                )
            
            # Case 2: There's a table pile - check if we can pick it up to complete sorting
            if table_pile and hand:
                # Try picking up the table pile (simulating "ALL" pickup with empty dealing)
                combined = list(hand) + list(table_pile)
                if is_sorted(combined):
                    # We can complete by picking up the table pile
                    # Create a final pass plan that represents picking up the table pile
                    final_plan = one_pass_greedy_distribution_with_table(
                        [], configs[0], "ALL", table_pile, table_pile_id
                    )
                    final_plan.next_deck = combined
                    
                    history = [deck[:]]
                    for plan in path:
                        history.append(plan.next_deck[:])
                    history.append(combined)
                    
                    explanations = []
                    for i, plan in enumerate(path):
                        config_desc = ", ".join([f"Pile {j+1}: {'top' if c == 'T' else 'bottom'}" 
                                                for j, c in enumerate(plan.config)])
                        pickup_desc = {
                            "ALL": "pickup both piles (P1, P2)",
                            "P1": "pickup P1 only (leave P2 on table)",
                            "P2": "pickup P2 only (leave P1 on table)",
                            "P2_FIRST": "pickup both piles (P2, P1)"
                        }.get(plan.pickup_strategy, plan.pickup_strategy)
                        explanations.append(f"Pass {i+1}: {config_desc}, {pickup_desc}")
                    explanations.append(f"Pass {len(path)+1}: pickup remaining table pile")
                    
                    return SortResult(
                        passes=len(path) + 1,
                        plans=path + [final_plan],
                        explanations=explanations,
                        history=history
                    )
            
            # Generate successors with greedy distribution and all pickup strategies
            for config in configs:
                for pickup_strategy in pickup_strategies:
                    plan = one_pass_greedy_distribution_with_table(
                        hand, config, pickup_strategy, table_pile, table_pile_id
                    )
                    
                    # New state after this pass
                    new_hand = tuple(plan.next_deck)
                    new_table_pile = plan.table_pile
                    new_table_pile_id = plan.table_pile_id
                    new_state = (new_hand, new_table_pile, new_table_pile_id)
                    
                    if new_state not in visited_global:
                        visited_global.add(new_state)
                        new_path = path + [plan]
                        next_beam.append((new_state, new_path))
        
        if not next_beam:
            break
        
        # Keep only top beam_width states based on combined heuristic
        # For states with table piles, we need to evaluate combined sortedness
        def score_state(state_path):
            state, path = state_path
            hand_tuple, table_pile, table_pile_id = state
            all_cards = list(hand_tuple) + list(table_pile) if table_pile else list(hand_tuple)
            return combined_heuristic(all_cards)
        
        next_beam_scored = [(score_state(item), item[0], item[1]) for item in next_beam]
        next_beam_scored.sort(reverse=True, key=lambda x: x[0])
        current_beam = [(state, path) for _, state, path in next_beam_scored[:beam_width]]
    
    # If no solution found, return best attempt
    if current_beam:
        def score_state_final(state_path):
            state, path = state_path
            hand_tuple, table_pile, table_pile_id = state
            all_cards = list(hand_tuple) + list(table_pile) if table_pile else list(hand_tuple)
            return combined_heuristic(all_cards)
        
        best_state, best_path = max(current_beam, key=score_state_final)
        history = [deck[:]]
        for plan in best_path:
            history.append(plan.next_deck[:])
        
        explanations = []
        for i, plan in enumerate(best_path):
            config_desc = ", ".join([f"Pile {j+1}: {'top' if c == 'T' else 'bottom'}" 
                                    for j, c in enumerate(plan.config)])
            pickup_desc = {
                "ALL": "pickup both piles (P1, P2)",
                "P1": "pickup P1 only (leave P2 on table)",
                "P2": "pickup P2 only (leave P1 on table)",
                "P2_FIRST": "pickup both piles (P2, P1)"
            }.get(plan.pickup_strategy, plan.pickup_strategy)
            explanations.append(f"Pass {i+1}: {config_desc}, {pickup_desc}")
        
        return SortResult(
            passes=len(best_path),
            plans=best_path,
            explanations=explanations,
            history=history
        )
    
    # Fallback: return empty result
    return SortResult(passes=0, plans=[], explanations=[], history=[deck[:]])


# ---------- User-friendly output functions ----------
def format_human_readable_plan(result: SortResult) -> List[str]:
    """
    Convert a SortResult into human-readable instructions.
    
    Args:
        result: The SortResult from optimal_sort
        
    Returns:
        List of strings with human-readable instructions
    """
    if result.passes == 0:
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
            print(f"[VERBOSE] SortResult: passes={result.passes}, history={result.history}")
        # Generate and print human-readable instructions
        instructions = format_human_readable_plan(result)
        print("\n".join(instructions))
        print(f"\nSorted in {result.passes} passes.")
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
        print(f"[VERBOSE] SortResult: passes={result.passes}, history={result.history}")
    expected_sorted = sorted(deck)
    if result.history and result.history[-1] != expected_sorted:
        return SortResult(
            passes=0,
            plans=[],
            explanations=["Solution produced but final state is not sorted; please adjust parameters."],
            history=[deck] + (result.history[-1:] if result.history else [])
        )
    return result
