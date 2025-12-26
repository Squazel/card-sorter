
"""
sort_logic.py

Consolidated card sorting algorithm for the Card Sorter project.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from growth_block_finder import find_biggest_block
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



# ---------- Main sorting algorithm ----------
def sequential_growth_sort(deck: List[int]) -> SortResult:
    """
    Sort using sequential growth method: repeatedly find the largest growth block,
    build the block by growing up and down in pile 2, put other cards in pile 1, then
    pick up pile 1 only; repeat until hand is sorted and can pick up both piles.
    """
    validate_input(deck)
    
    if len(deck) <= 1:
        return SortResult(passes=0, plans=[], explanations=[], history=[deck[:]])
    
    if is_sorted(deck):
        return SortResult(passes=0, plans=[], explanations=[], history=[deck[:]])
    
    plans = []
    history = [deck[:]]
    current = deck[:]
    table_pile = ()
    table_pile_id = None
    passes = 0
    
    while True:
        if not current and not table_pile:
            break
        
        if not current and table_pile:
            # Pick up table pile
            current = list(table_pile)
            table_pile = ()
            table_pile_id = None
            history.append(current[:])
            continue
        
        # Find the largest growth block
        chain = find_biggest_block(current)
        orientation2 = 'B' if chain == sorted(chain) else 'T'
        
        if not chain or len(chain) < 2:
            # No useful chain, check if we can finish
            if table_pile and is_sorted(current + list(table_pile)):
                # Pick up both piles
                final_deck = current + list(table_pile)
                history.append(final_deck[:])
                
                # Create final plan
                final_plan = PassPlan(
                    config=('B', 'B'),
                    moves=[],
                    next_deck=final_deck,
                    piles_snapshot={},
                    pickup_strategy="ALL",
                    table_pile=(),
                    table_pile_id=None
                )
                plans.append(final_plan)
                passes += 1
                break
            else:
                raise ValueError("Unable to sort the deck with sequential growth method.")
        
        # Place chain to pile 2 with determined orientation
        pile2 = chain[:]
        
        # Other cards to pile 1 with 'B' orientation
        other_cards = [c for c in current if c not in chain]
        pile1 = other_cards[:]
        orientation1 = 'B'
        
        # Create moves
        moves = []
        # Pile 1 moves
        for i, card in enumerate(pile1):
            if i == 0:
                moves.append(Move(card, "P1"))
            else:
                moves.append(Move(card, f"P1-{orientation1}"))
        # Pile 2 moves
        for i, card in enumerate(pile2):
            if i == 0:
                moves.append(Move(card, "P2"))
            else:
                moves.append(Move(card, f"P2-{orientation2}"))
        
        # Build piles for pickup
        piles_for_pickup = {
            f"P1-{orientation1}": materialize_pile_for_pickup(orientation1, pile1),
            f"P2-{orientation2}": materialize_pile_for_pickup(orientation2, pile2)
        }
        
        # Pick up pile 1 only
        next_deck = piles_for_pickup[f"P1-{orientation1}"]
        table_pile = tuple(piles_for_pickup[f"P2-{orientation2}"])
        table_pile_id = 2
        
        # Create plan
        plan = PassPlan(
            config=(orientation1, orientation2),
            moves=moves,
            next_deck=next_deck,
            piles_snapshot=piles_for_pickup,
            pickup_strategy="P1",
            table_pile=table_pile,
            table_pile_id=table_pile_id
        )
        plans.append(plan)
        
        current = next_deck
        history.append(current[:])
        passes += 1
    
    return SortResult(passes=passes, plans=plans, explanations=[], history=history)


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
        result = sequential_growth_sort(deck)
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
        result = sequential_growth_sort(deck)
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
