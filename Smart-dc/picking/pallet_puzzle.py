"""
Pallet rearrangement in a dense storage block (the tiles / 8-puzzle problem).

A puzzle-based storage block holds 8 pallets in a 3 x 3 grid with one empty
slot. A pallet can only slide into the empty slot next to it. The goal is
to reach a target layout, for example pallets 1-8 in dispatch order with
the empty slot at the exit corner, in the fewest slides.

State: a tuple of 9 numbers read row by row, 0 = the empty slot.
Search: A* with two admissible heuristics, compared on nodes expanded:
  - misplaced pallets: how many pallets are not in their target slot
  - Manhattan distance: total grid steps each pallet is from its target

Syllabus topics: Module 2 - Tiles problem; Module 1 - Heuristic, A* search.

Run on its own:   python picking/pallet_puzzle.py
"""

import heapq
from itertools import count

SIZE = 3
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

# Moving the empty slot in a direction = the neighbouring pallet slides the
# opposite way. (row change, column change, how the pallet moves)
DIRECTIONS = [(-1, 0, "down"), (1, 0, "up"), (0, -1, "right"), (0, 1, "left")]


def is_solvable(state, goal=GOAL):
    """On a 3 x 3 board a layout is reachable only if its inversion parity
    matches the goal's."""
    def inversions(s):
        tiles = [t for t in s if t != 0]
        return sum(1 for i in range(len(tiles)) for j in range(i + 1, len(tiles))
                   if tiles[i] > tiles[j])
    return inversions(state) % 2 == inversions(goal) % 2


def misplaced(state, goal=GOAL):
    return sum(1 for s, g in zip(state, goal) if s != 0 and s != g)


def manhattan(state, goal=GOAL):
    where = {tile: divmod(i, SIZE) for i, tile in enumerate(goal)}
    total = 0
    for i, tile in enumerate(state):
        if tile:
            r, c = divmod(i, SIZE)
            gr, gc = where[tile]
            total += abs(r - gr) + abs(c - gc)
    return total


HEURISTICS = {"misplaced pallets": misplaced, "Manhattan distance": manhattan}


def neighbours(state):
    """Every layout one slide away, with a description of the slide."""
    empty = state.index(0)
    r, c = divmod(empty, SIZE)
    for dr, dc, pallet_moves in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < SIZE and 0 <= nc < SIZE:
            j = nr * SIZE + nc
            new = list(state)
            new[empty], new[j] = new[j], new[empty]
            yield tuple(new), f"pallet {state[j]} slides {pallet_moves}"


def solve(start, goal=GOAL, heuristic=manhattan):
    """A* search. Returns (moves, nodes_expanded); moves is None if unsolvable."""
    if sorted(start) != list(range(SIZE * SIZE)):
        raise ValueError("A layout needs pallets 1-8 and one empty slot (0)")
    if not is_solvable(start, goal):
        return None, 0

    tie = count()
    open_list = [(heuristic(start, goal), next(tie), start)]
    g_cost = {start: 0}
    parent = {start: (None, None)}
    expanded = 0

    while open_list:
        _, _, state = heapq.heappop(open_list)
        if state == goal:
            moves = []
            while parent[state][0] is not None:
                state, move = parent[state]
                moves.append(move)
            moves.reverse()
            return moves, expanded
        expanded += 1
        for nxt, move in neighbours(state):
            new_g = g_cost[state] + 1
            if nxt not in g_cost or new_g < g_cost[nxt]:
                g_cost[nxt] = new_g
                parent[nxt] = (state, move)
                heapq.heappush(open_list, (new_g + heuristic(nxt, goal), next(tie), nxt))
    return None, expanded


def show(state):
    rows = []
    for r in range(SIZE):
        cells = state[r * SIZE:(r + 1) * SIZE]
        rows.append(" ".join(f"P{t}" if t else "--" for t in cells))
    return "\n".join(rows)


DEMO_START = (4, 1, 2,
              0, 8, 7,
              6, 3, 5)

# Extra layouts for the heuristic comparison table (all solvable).
COMPARISON_LAYOUTS = [
    (5, 1, 3, 4, 0, 2, 7, 8, 6),
    (8, 1, 3, 4, 0, 2, 7, 6, 5),
    DEMO_START,
    (5, 2, 8, 4, 1, 7, 0, 3, 6),
]


def run_demo():
    print("Storage block now:          Target layout:")
    for now, target in zip(show(DEMO_START).split("\n"), show(GOAL).split("\n")):
        print(f"{now:<28}{target}")
    print()

    results = {}
    for name, h in HEURISTICS.items():
        results[name] = solve(DEMO_START, GOAL, h)

    moves, _ = results["Manhattan distance"]
    print(f"Fewest slides: {len(moves)}")
    for step, move in enumerate(moves, 1):
        print(f"  {step:>2}. {move}")
    print()
    print("Nodes expanded by A* with each heuristic (both find the fewest slides):")
    print(f"{'Layout':<30}{'Slides':>7}{'Misplaced':>11}{'Manhattan':>11}")
    for layout in COMPARISON_LAYOUTS:
        m1, n1 = solve(layout, GOAL, misplaced)
        m2, n2 = solve(layout, GOAL, manhattan)
        assert len(m1) == len(m2)
        print(f"{str(layout):<30}{len(m2):>7}{n1:>11}{n2:>11}")

    unsolvable = (1, 2, 3, 4, 5, 6, 8, 7, 0)
    print()
    print(f"Layout {unsolvable} solvable? {is_solvable(unsolvable)} "
          f"(two pallets swapped can never be fixed by sliding)")


if __name__ == "__main__":
    run_demo()
