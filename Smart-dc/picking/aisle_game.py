"""
Two robots competing for pallets in one aisle (game playing).

Pallets stand in a single line along an aisle, and each pallet carries
priority points. Robot A and robot B take turns; on its turn a robot must
take the pallet at either open end of the aisle. Robot A wants to finish
with as many more points than robot B as possible, and plans for the
worst case: it assumes robot B also plays its best.

Game value = A's points - B's points.
  - Minimax searches the full game tree.
  - Alpha-beta pruning returns the same value while skipping branches that
    cannot change the decision.

Syllabus topics: Module 2 - Game playing: minimax, alpha-beta pruning.

Run on its own:   python picking/aisle_game.py
"""

import random


class Counter:
    def __init__(self):
        self.nodes = 0


def minimax(pallets, lo, hi, robot_a_turn, counter):
    """Value of the game when pallets[lo..hi] are left."""
    counter.nodes += 1
    if lo > hi:
        return 0
    take_left = pallets[lo] if robot_a_turn else -pallets[lo]
    take_right = pallets[hi] if robot_a_turn else -pallets[hi]
    left = take_left + minimax(pallets, lo + 1, hi, not robot_a_turn, counter)
    right = take_right + minimax(pallets, lo, hi - 1, not robot_a_turn, counter)
    return max(left, right) if robot_a_turn else min(left, right)


def alphabeta(pallets, lo, hi, robot_a_turn, counter,
              alpha=float("-inf"), beta=float("inf"), so_far=0):
    """Same value as minimax, with pruning.

    so_far = points difference collected so far on this branch. Carrying
    it down makes leaf values absolute, so alpha and beta compare like
    with like at every depth.
    """
    counter.nodes += 1
    if lo > hi:
        return so_far

    sign = 1 if robot_a_turn else -1
    options = [(lo + 1, hi, pallets[lo]), (lo, hi - 1, pallets[hi])]

    if robot_a_turn:
        value = float("-inf")
        for new_lo, new_hi, points in options:
            value = max(value, alphabeta(pallets, new_lo, new_hi, False, counter,
                                         alpha, beta, so_far + sign * points))
            alpha = max(alpha, value)
            if alpha >= beta:          # robot B already has a better branch
                break
        return value

    value = float("inf")
    for new_lo, new_hi, points in options:
        value = min(value, alphabeta(pallets, new_lo, new_hi, True, counter,
                                     alpha, beta, so_far + sign * points))
        beta = min(beta, value)
        if alpha >= beta:              # robot A already has a better branch
            break
    return value


def best_move(pallets, lo, hi, robot_a_turn):
    """'left' or 'right' for the robot to move, by alpha-beta search."""
    sign = 1 if robot_a_turn else -1
    scores = {}
    for side, (new_lo, new_hi, points) in {
            "left": (lo + 1, hi, pallets[lo]),
            "right": (lo, hi - 1, pallets[hi])}.items():
        scores[side] = alphabeta(pallets, new_lo, new_hi, not robot_a_turn,
                                 Counter(), so_far=sign * points)
    if robot_a_turn:
        return max(scores, key=lambda s: (scores[s], s == "left"))
    return min(scores, key=lambda s: (scores[s], s != "left"))


def play(pallets):
    """Play the whole game with both robots using best moves."""
    lo, hi = 0, len(pallets) - 1
    robot_a_turn = True
    points = {"A": 0, "B": 0}
    moves = []
    while lo <= hi:
        side = best_move(pallets, lo, hi, robot_a_turn)
        taken = pallets[lo] if side == "left" else pallets[hi]
        if side == "left":
            lo += 1
        else:
            hi -= 1
        robot = "A" if robot_a_turn else "B"
        points[robot] += taken
        moves.append((robot, side, taken))
        robot_a_turn = not robot_a_turn
    return moves, points


def greedy_play(pallets):
    """Robot A always takes the bigger end; robot B still plays its best."""
    lo, hi = 0, len(pallets) - 1
    robot_a_turn = True
    points = {"A": 0, "B": 0}
    while lo <= hi:
        if robot_a_turn:
            side = "left" if pallets[lo] >= pallets[hi] else "right"
        else:
            side = best_move(pallets, lo, hi, False)
        taken = pallets[lo] if side == "left" else pallets[hi]
        if side == "left":
            lo += 1
        else:
            hi -= 1
        points["A" if robot_a_turn else "B"] += taken
        robot_a_turn = not robot_a_turn
    return points


def compare(pallets):
    """(value, minimax nodes, alpha-beta nodes) for one aisle."""
    c1, c2 = Counter(), Counter()
    v1 = minimax(pallets, 0, len(pallets) - 1, True, c1)
    v2 = alphabeta(pallets, 0, len(pallets) - 1, True, c2)
    if v1 != v2:
        raise AssertionError("alpha-beta must return the minimax value")
    return v1, c1.nodes, c2.nodes


DEMO_AISLE = [3, 9, 1, 2, 7, 15, 4, 6]


def run_demo():
    print(f"Pallet points along the aisle: {DEMO_AISLE}")
    print()
    moves, points = play(DEMO_AISLE)
    print("Both robots play their best:")
    for turn, (robot, side, taken) in enumerate(moves, 1):
        print(f"  {turn}. robot {robot} takes the {side} pallet "
              f"({taken} point{'' if taken == 1 else 's'})")
    print(f"Result: A {points['A']}, B {points['B']}, "
          f"difference {points['A'] - points['B']}")

    greedy = greedy_play(DEMO_AISLE)
    print(f"If robot A just grabs the bigger end each time: "
          f"A {greedy['A']}, B {greedy['B']}, difference {greedy['A'] - greedy['B']}")
    print()

    rng = random.Random(7)
    aisles = [DEMO_AISLE] + [[rng.randint(1, 20) for _ in range(n)] for n in (10, 12, 14, 16)]
    print("Nodes explored, same game value from both:")
    print(f"{'Pallets':>8}{'Value':>8}{'Minimax':>10}{'Alpha-beta':>12}{'Saved':>8}")
    for aisle in aisles:
        value, mm, ab = compare(aisle)
        print(f"{len(aisle):>8}{value:>8}{mm:>10}{ab:>12}{(mm - ab) / mm:>8.0%}")


if __name__ == "__main__":
    run_demo()
