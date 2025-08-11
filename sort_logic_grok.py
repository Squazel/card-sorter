import collections
from typing import List, Tuple

def generate_moves():
    moves = []
    # 1 pile, top (reverse)
    moves.append((1, [True], "1 pile, Pile 0: top"))
    # 2 piles, all combinations
    for rev0 in [False, True]:
        for rev1 in [False, True]:
            desc = f"2 piles, Pile 0: {'top' if rev0 else 'bottom'}, Pile 1: {'top' if rev1 else 'bottom'}"
            moves.append((2, [rev0, rev1], desc))
    return moves

def apply_move(perm: List[int], num_piles: int, rev_list: List[bool]) -> List[int]:
    if num_piles == 1:
        if rev_list[0]:
            return perm[::-1]
        else:
            return perm[:]
    else:  # 2 piles
        pile0 = perm[::2]
        if rev_list[0]:
            pile0 = pile0[::-1]
        pile1 = perm[1::2]
        if rev_list[1]:
            pile1 = pile1[::-1]
        return pile0 + pile1

def find_min_iterations(perm: List[int]) -> Tuple[int, List[str], List[List[int]]]:
    n = len(perm)
    start = tuple(perm)
    goal = tuple(range(1, n + 1))
    if start == goal:
        return 0, [], [perm[:]]

    moves = generate_moves()
    visited = set([start])
    came_from = {start: (None, None)}
    queue = collections.deque([start])

    found = False
    while queue:
        current = queue.popleft()
        if current == goal:
            found = True
            break
        curr_list = list(current)
        for num_piles, rev_list, desc in moves:
            new_list = apply_move(curr_list, num_piles, rev_list)
            new_tuple = tuple(new_list)
            if new_tuple not in visited:
                visited.add(new_tuple)
                came_from[new_tuple] = (current, desc)
                queue.append(new_tuple)

    if not found:
        raise ValueError("Unable to sort the permutation with these operations.")

    # Reconstruct path
    path = []
    history = []
    current = goal
    while current != start:
        prev, move_desc = came_from[current]
        if move_desc is not None:  # Skip initial None
            path.append(move_desc)
            history.append(list(current))
        current = prev
    path.reverse()
    history.reverse()
    history.insert(0, perm[:])  # Add starting permutation

    return len(path), path, history

def main():
    # Input validation
    try:
        perm = input("Enter permutation of numbers 1 to 13 (space-separated, e.g., 13 12 11 ... 1): ").strip().split()
        perm = [int(x) for x in perm]
        if sorted(perm) != list(range(1, 14)):
            raise ValueError("Permutation must contain exactly the numbers 1 to 13.")
        
        # Find minimum iterations and path
        iterations, moves, history = find_min_iterations(perm)
        
        # Output results
        print(f"Minimum iterations needed: {iterations}")
        print(f"\nSorted permutation: {history[-1]}")
        print("\nOperations and intermediate permutations:")
        print(f"Iteration 0: {history[0]} (initial)")
        for i in range(iterations):
            print(f"Operation {i+1}: {moves[i]}")
            print(f"Iteration {i+1}: {history[i+1]}")
            
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()