# Card Sorting Algorithm

## Overview

This document describes the algorithm used to sort a hand of cards through a series of physical manipulations, optimized for use cases where an assistant helps sort cards without seeing them (e.g., in Bridge).

## Problem Statement

Given:
- An initial hand of 13 cards in a known order (to the system, not the user)
- Two piles (Pile 1 and Pile 2) into which cards can be dealt
- The ability to place cards on Top (T) or Bottom (B) of each pile
- The ability to pick up one or both piles after dealing

Find: The minimum number of iterations needed to sort the cards into ascending order.

## User Actions Model

### Dealing Phase
1. User starts with all 13 cards face-down in their hand
2. For each card (from top of hand), the system instructs which pile to deal to:
   - First card to each pile: must go on top (pile is empty)
   - Subsequent cards: can be placed on Top (T) or Bottom (B) of the pile

### Pickup Phase
After all cards are dealt, the system instructs which pile(s) to pick up:
- **Option 1**: Pick up Pile 1 only (Pile 2 remains on table)
- **Option 2**: Pick up Pile 2 only (Pile 1 remains on table)
- **Option 3**: Pick up both piles, with Pile 1 on top of Pile 2
- **Option 4**: Pick up both piles, with Pile 2 on top of Pile 1

Any pile left on the table remains in place for the next iteration.

### Iteration
The process repeats until all cards are sorted.

## Algorithmic Approach

### Current Implementation: Configuration-Based BFS

The current implementation uses a **Breadth-First Search (BFS)** over deck states with a fixed distribution strategy:

#### Key Components

1. **State Representation**: Each state is a tuple representing the current order of all cards in hand
   - **Note**: The current implementation assumes all cards are picked up after each iteration
   - The new action model allows at most one pile to persist on the table, which would require an extended state representation: `(hand_cards, table_pile_cards, which_pile)` where `which_pile` indicates whether it's pile 1 or pile 2 (or None if no pile on table)

2. **Configuration Space**: For 2 piles, we have 4 configurations:
   - (T, T): Both piles with top placement
   - (T, B): Pile 1 top, Pile 2 bottom
   - (B, T): Pile 1 bottom, Pile 2 top
   - (B, B): Both piles with bottom placement

3. **Distribution Strategy**: Cards are distributed round-robin:
   - Card at index i goes to pile (i mod num_piles)
   - This is deterministic and reduces the search space
   - **Limitation**: The new action model allows free choice of which pile each card goes to, which could potentially lead to better solutions

4. **Pickup Strategy**: Piles are always picked up in fixed order (Pile 1, then Pile 2)
   - **Limitation**: The new action model allows picking up only P1, only P2, both (P1+P2), or both (P2+P1)
   - This flexibility could reduce the number of iterations needed

#### BFS Guarantees

The BFS approach guarantees:
- **Optimality**: First solution found has minimum number of iterations (for the given distribution strategy)
- **Completeness**: Will find a solution if one exists
- **Correctness**: Always produces sorted output when possible

### Theoretical Foundations

#### Relation to Sorting Networks

The pile-based sorting approach is related to [**sorting networks**](https://en.wikipedia.org/wiki/Sorting_network), particularly:

- **Comparator Networks**: Each pile acts as a filter that orders elements based on the T/B configuration
- **Parallel Sorting**: Multiple piles allow parallel processing of cards
- **Oblivious Sorting**: The sequence of operations is data-independent (determined before execution)

#### Relation to Patience Sorting

The algorithm has similarities to [**Patience Sorting**](https://en.wikipedia.org/wiki/Patience_sorting):

- Cards are dealt into piles following specific rules
- Piles maintain certain ordering properties
- Piles are recombined to produce the final sorted sequence

Key differences:
- Patience sorting uses a greedy strategy (place card on leftmost valid pile)
- Our algorithm explores multiple configurations systematically
- Patience sorting finds longest increasing subsequence; we optimize for minimum iterations

#### Relation to Merge-Based Algorithms

The pile operations resemble **merge-based sorting**:

- **Top placement (T)**: Reverses the order of cards in a pile (similar to a stack)
- **Bottom placement (B)**: Preserves the order of cards in a pile (similar to a queue)
- **Recombination**: Merging piles creates the next iteration's deck

This is related to [**Merge Sort**](https://en.wikipedia.org/wiki/Merge_sort), which also recursively divides and recombines data.

### Complexity Analysis

#### Current Implementation

With the round-robin distribution constraint:

- **State Space**: O(n!) possible deck orderings, but BFS explores only reachable states
- **Per-State Branching**: 4 configurations for 2 piles (constant)
- **Time Complexity**: O(n! × c) where c = 4 is the number of configurations
- **Space Complexity**: O(n! × n) for visited states storage

For n=13 cards:
- Theoretical maximum states: 13! ≈ 6.2 billion
- Practical states explored: Much fewer due to early termination when sorted

#### Full Distribution Freedom

If we explore all possible distributions (not just round-robin):

- **Distribution Choices**: 2^13 ways to assign 13 cards to 2 piles
- **Configuration Choices**: 4 T/B combinations per distribution
- **Pickup Choices**: 4 ways to pick up piles (P1, P2, P1+P2, P2+P1)
- **Total per iteration**: 2^13 × 4 × 4 = 262,144 possibilities

This becomes computationally expensive for a full BFS across all iterations.

### Optimization Strategies

To handle the expanded action space efficiently:

1. **Heuristic Search**: Use A* with sortedness heuristic instead of pure BFS
2. **Pruning**: Eliminate symmetric or dominated configurations early
3. **Bounded Search**: Limit maximum iterations or search depth
4. **Greedy Approximation**: Use heuristics to narrow distribution choices

## Practical Constraints

For the Bridge use case (13 cards):

- **Target**: Find solution in < 1 second on typical hardware
- **Memory**: Limit visited states to reasonable size (< 100MB)
- **Iterations**: Prefer solutions with ≤ 5 iterations for user experience

### Current Performance Limitations

**Important**: The current BFS implementation has **practical limits** on deck size:

- **Efficient range**: 3-7 cards (completes in under 1 second)
- **Practical limit**: 8-10 cards (may take several seconds)
- **Known issue**: 11-13 cards can take **minutes to hours** depending on initial order

For 13 cards:
- Theoretical state space: 13! = 6,227,020,800 permutations
- BFS must explore many states before finding a solution
- Some orderings may exhaust memory or take prohibitively long

**Workaround**: For 13-card hands that are slow, consider:
1. Using fewer piles (but this may not find a solution)
2. Using a heuristic approach instead of exhaustive BFS
3. Implementing the enhanced state representation with pile persistence (future work)

This limitation is documented in the "Implementation Gap" section below.

## Implementation Notes

The current implementation in `sort_logic.py`:

1. **Validates** input to ensure distinct positive integers
2. **Uses BFS** to explore configuration space systematically  
3. **Caps piles** at 2 for reliability and performance
4. **Tracks states** to avoid redundant exploration
5. **Returns** detailed step-by-step instructions for the user

### Implementation Gap

The current implementation provides optimal solutions **within its constraints** (round-robin distribution, fixed pickup order). However, the new action model described above allows additional flexibility that could potentially reduce the number of iterations:

1. **Free distribution choice**: Instead of round-robin, allowing each card to go to any pile
2. **Flexible pickup strategy**: Allowing piles to be left on the table across iterations
3. **Persistent piles**: Modeling cards that remain on the table and accumulate cards in subsequent iterations

Implementing these enhancements would require:
- Expanding the state representation from `(cards_in_hand)` to `(cards_in_hand, table_pile_cards, which_pile)` where at most one pile persists on the table
- Modifying the BFS to explore different distribution patterns (not just round-robin)
- Adding pickup strategy as a dimension in the search space (4 options: P1 only, P2 only, both in either order)
- Handling the complexity of dealing cards onto a pile that already contains cards from a previous iteration

The trade-off is between optimality and computational complexity. The current implementation favors tractability and reliably produces good (though not necessarily globally optimal) solutions for 13-card hands.

## Future Enhancements

Possible improvements to explore:

### High Priority: Close Implementation Gap

To fully support the new action model described in this document:

1. **Enhanced State Representation**
   - Change from `tuple[int, ...]` to `(hand: tuple, table_pile: tuple, pile_id: int | None)`
   - Note: At most one pile persists on the table between iterations (never both)
   - Update BFS to handle this extended state space
   - Modify goal condition to check if all cards are sorted (considering hand + any pile on table)

2. **Flexible Distribution Search**
   - Implement heuristic-guided distribution (e.g., greedy based on sortedness)
   - Use A* instead of BFS with a good heuristic function
   - Consider beam search to limit explored distributions to top-k candidates

3. **Pickup Strategy Exploration**
   - Add pickup strategy as a parameter: `'P1'`, `'P2'`, `'P1+P2'`, or `'P2+P1'`
   - Explore all 4 pickup strategies at each BFS level
   - Model the effect of leaving piles on the table

4. **Dealing onto Existing Pile**
   - Handle case where one pile already has cards from previous iteration
   - Update T/B placement logic to work with a non-empty starting pile
   - Maintain proper ordering as cards accumulate on the persisting pile

### Medium Priority: Performance Optimization

5. **Parallel Search**: Use multiple cores to explore configuration space faster
6. **Dynamic Programming**: Memoize subproblems for efficiency
7. **Pruning Strategies**: Eliminate symmetric or dominated configurations early

### Lower Priority: Advanced Techniques

8. **Machine Learning**: Train models to predict good distributions for given initial orders
9. **Pattern Recognition**: Identify common sorting patterns and optimize for them
10. **Adaptive Heuristics**: Learn from previous sorts to improve future performance

## References

- **Sorting Networks**: Knuth, Donald E. (1998). "The Art of Computer Programming, Volume 3: Sorting and Searching"
- **Patience Sorting**: Mallows, C. L. (1963). "Problem 62-2, patience sorting"
- **BFS Optimality**: Cormen et al. (2009). "Introduction to Algorithms, 3rd Edition"
- **A* Search**: Hart, P. E.; Nilsson, N. J.; Raphael, B. (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths"

## Validation

The algorithm's correctness is validated through:

- **Unit Tests**: Comprehensive test suite covering edge cases
- **Optimality Tests**: Brute-force verification for small cases
- **Performance Tests**: Benchmarking on typical Bridge hands
- **Integration Tests**: End-to-end validation with card representations

See `tests/` directory for complete test coverage.
