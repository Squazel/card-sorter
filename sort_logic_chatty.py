#!/usr/bin/env python3
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# ---------- Core types ----------
@dataclass
class Move:
    card: int
    where: str  # 'P1-T','P1-B','P2-T','P2-B','L' (Leftover)

@dataclass
class PassPlan:
    config: Tuple[str, ...]         # e.g. ('B',) or ('T','B') with 'T' = top (reverse), 'B' = bottom (preserve)
    moves: List[Move]               # per card decisions in scan order
    next_deck: List[int]            # deck after recombining
    piles_snapshot: Dict[str, List[int]]  # recorded as they will appear when picked up into next_deck

# ---------- Helpers ----------
def is_sorted(deck: List[int]) -> bool:
    return all(deck[i] <= deck[i+1] for i in range(len(deck)-1))

def can_place_on(pile_kind: str, last: Optional[int], x: int) -> bool:
    if last is None:
        return True
    return (x > last) if pile_kind == 'B' else (x < last)  # B=increasing, T=decreasing while scanning

def materialise_pile_for_pickup(pile_kind: str, seq: List[int]) -> List[int]:
    # When you "pick up" the pile for recombination:
    # - 'B' preserves order (queue-like), 'T' reverses (stack-like)
    return seq if pile_kind == 'B' else list(reversed(seq))

def one_pass(deck: List[int], config: Tuple[str, ...]) -> PassPlan:
    """
    Execute one pass with fixed pile-orientations given by config.
    config = ('B',)            -> one pile, bottom (preserve)
           = ('T',)            -> one pile, top (reverse)
           = ('T','T'), ('T','B'), ('B','B') -> two piles
    Returns the full PassPlan including per-card move tracing.
    """
    assert all(c in ('T','B') for c in config)
    r = len(config)
    piles: List[List[int]] = [[] for _ in range(r)]
    last_vals: List[Optional[int]] = [None]*r
    moves: List[Move] = []
    leftovers: List[int] = []

    for x in deck:
        placed = False
        # Try Pile 1 then Pile 2 (if exists)
        for j in range(r):
            if can_place_on(config[j], last_vals[j], x):
                piles[j].append(x)
                last_vals[j] = x
                moves.append(Move(x, f"P{j+1}-{'T' if config[j]=='T' else 'B'}"))
                placed = True
                break
        if not placed:
            leftovers.append(x)
            moves.append(Move(x, "L"))

    # Build next deck: leftovers on top, then P2, then P1 (customise if needed)
    piles_for_pickup = {
        f"P{j+1}-{config[j]}": materialise_pile_for_pickup(config[j], piles[j])
        for j in range(r)
    }
    next_deck = leftovers[:]
    if r == 2:
        next_deck.extend(piles_for_pickup["P2-"+config[1]])
    if r >= 1:
        next_deck.extend(piles_for_pickup["P1-"+config[0]])

    return PassPlan(config=config, moves=moves, next_deck=next_deck, piles_snapshot=piles_for_pickup)

# ---------- Optimal planner via BFS over passes ----------
def optimal_sort(deck: List[int], piles: int) -> Tuple[List[PassPlan], int]:
    """
    Find a minimum number of passes to sort 'deck' under the model described.
    'piles' is 1 or 2. For piles=1, allowed configs: ('B',), ('T',).
    For piles=2, allowed configs: ('T','T'), ('T','B'), ('B','B').
    Returns (sequence_of_passes, pass_count).
    """
    if sorted(deck) != list(range(1, len(deck)+1)):
        raise ValueError("Deck must be a permutation of 1..n (e.g., 1..13).")
    if piles not in (1, 2):
        raise ValueError("This solver currently supports piles ∈ {1, 2}.")

    start = tuple(deck)
    if is_sorted(deck):
        return [], 0

    if piles == 1:
        configs = [("B",), ("T",)]
    else:
        configs = [("T","T"), ("T","B"), ("B","B")]

    # BFS over decks (state graph), edges labelled by PassPlan
    q = deque()
    q.append(start)
    parent: Dict[Tuple[int, ...], Tuple[Tuple[int, ...], PassPlan]] = {}
    seen = {start}

    while q:
        state = q.popleft()
        if is_sorted(list(state)):
            # Reconstruct path
            plan_seq: List[PassPlan] = []
            cur = state
            while cur != start:
                prev, plan = parent[cur]
                plan_seq.append(plan)
                cur = prev
            plan_seq.reverse()
            return plan_seq, len(plan_seq)

        # Expand: try each allowed config deterministically
        for cfg in configs:
            plan = one_pass(list(state), cfg)
            nxt = tuple(plan.next_deck)
            if nxt not in seen:
                seen.add(nxt)
                parent[nxt] = (state, plan)
                q.append(nxt)

    # Should never happen for finite state space
    raise RuntimeError("No solution found (unexpected).")

# ---------- Pretty printing ----------
def print_solution(deck: List[int], piles: int) -> None:
    plans, count = optimal_sort(deck, piles)
    print(f"Start: {deck}")
    for t, plan in enumerate(plans, 1):
        label = "+".join(plan.config)
        print(f"\nPass {t}   (config: {label}   where T=top/reverse, B=bottom/preserve)")
        # Show how the piles will appear when picked up:
        for k, v in plan.piles_snapshot.items():
            print(f"  {k} pickup order: {v}")
        # Moves per card:
        mv = ", ".join(f"{m.card}->{m.where}" for m in plan.moves)
        print(f"  Moves: {mv}")
        print(f"  Next deck: {plan.next_deck}")
    print(f"\nSorted in {count} passes.")

# ---------- Example ----------
if __name__ == "__main__":
    # Example inputs:
    deck = [2, 11, 10, 6, 12, 3, 4, 8, 9, 5, 1, 13, 7]  # any permutation of 1..n
    # Try with one pile (top-or-bottom) OR two piles (top/top, top/bottom, bottom/bottom):
    print("=== One pile (choose T or B each pass) ===")
    print_solution(deck, piles=1)
    print("\n=== Two piles (choose TT/TB/BB each pass) ===")
    print_solution(deck, piles=2)
