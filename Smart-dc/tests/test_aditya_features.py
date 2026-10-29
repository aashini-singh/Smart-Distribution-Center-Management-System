"""
Tests for Aditya's features. Each result is checked against a separate,
simpler method (Dijkstra, brute force, BFS, plain minimax).

Run from the Smart-dc folder:
    python -m unittest tests.test_aditya_features -v
"""

import heapq
import random
import sys
import unittest
from collections import deque
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from picking.route_costs import GraphRouter, graph_astar
from picking.tsp_exact import exact_picking_order
from picking.ao_star import (AndOrGraph, ao_star, exhaustive_cost,
                             lower_bounds, build_demo)
from picking import pallet_puzzle as pp
from picking import aisle_game as game


def dijkstra(graph, start):
    dist = {start: 0}
    heap = [(0, start)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        for nb, cost in graph.get_neighbors(node):
            if d + cost < dist.get(nb, float("inf")):
                dist[nb] = d + cost
                heapq.heappush(heap, (d + cost, nb))
    return dist


class RouteCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = GraphRouter.from_file()
        cls.graph = cls.router.graph

    def test_astar_matches_dijkstra_on_every_pair(self):
        for start in self.graph.locations:
            truth = dijkstra(self.graph, start)
            for goal in self.graph.locations:
                self.assertEqual(self.router.distance(start, goal), truth[goal],
                                 f"{start} -> {goal}")

    def test_heuristic_never_overestimates(self):
        for goal in self.graph.locations:
            truth = dijkstra(self.graph, goal)
            for node in self.graph.locations:
                g, n = self.graph.get_location(goal), self.graph.get_location(node)
                h = self.router.scale * (abs(g.x - n.x) + abs(g.y - n.y))
                self.assertLessEqual(h, truth[node] + 1e-9)

    def test_path_cost_adds_up(self):
        result = graph_astar(self.graph, "P1", "L21")
        total = sum(self.graph.get_cost(a, b)
                    for a, b in zip(result["path"], result["path"][1:]))
        self.assertEqual(total, result["cost"])

    def test_exact_tsp_on_graph_matches_brute_force(self):
        stops = ["S9", "L04", "L10", "S1", "L08"]
        plan = exact_picking_order(stops, self.router.distance)
        best = min(
            sum(self.router.distance(a, b)
                for a, b in zip(("P1",) + p + ("D1",), p + ("D1",)))
            for p in permutations(stops))
        self.assertEqual(plan.cost, best)


class AOStarTests(unittest.TestCase):
    def test_demo_is_cheapest_and_skips_the_transfer(self):
        graph, bounds = build_demo(GraphRouter.from_file())
        result = ao_star(graph, "ORD201", bounds)
        self.assertEqual(result.cost, exhaustive_cost(graph, "ORD201"))
        self.assertNotIn("Monitor by partner-DC transfer", result.expansion_order)
        self.assertLess(len(result.expansion_order), result.total_nodes)

    def test_random_and_or_graphs(self):
        rng = random.Random(3)
        for trial in range(300):
            g = AndOrGraph()
            counter = [0]

            def build(name, depth):
                if depth == 0 or rng.random() < 0.3:
                    g.add_leaf(name, rng.randint(1, 50))
                    return
                for _ in range(rng.randint(1, 3)):
                    counter[0] += 1
                    k = rng.randint(1, 3)
                    kids = [f"n{counter[0]}_{i}" for i in range(k)]
                    g.add_and(name, *kids, cost=rng.randint(0, 5))
                    for kid in kids:
                        build(kid, depth - 1)

            build("root", 3)
            truth = exhaustive_cost(g, "root")
            leaf_bounds = {n: c * rng.random() for n, c in g.leaf_cost.items()}
            self.assertEqual(ao_star(g, "root").cost, truth, trial)
            self.assertAlmostEqual(
                ao_star(g, "root", lower_bounds(g, leaf_bounds)).cost, truth)


class PalletPuzzleTests(unittest.TestCase):
    def bfs_length(self, start):
        frontier, seen = deque([(start, 0)]), {start}
        while frontier:
            state, d = frontier.popleft()
            if state == pp.GOAL:
                return d
            for nxt, _ in pp.neighbours(state):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append((nxt, d + 1))
        return None

    def replay(self, start, moves):
        state = start
        for move in moves:
            matches = [n for n, m in pp.neighbours(state) if m == move]
            self.assertEqual(len(matches), 1, move)
            state = matches[0]
        return state

    def test_both_heuristics_find_the_fewest_slides(self):
        for layout in pp.COMPARISON_LAYOUTS[:3]:
            optimum = self.bfs_length(layout)
            for h in (pp.misplaced, pp.manhattan):
                moves, _ = pp.solve(layout, pp.GOAL, h)
                self.assertEqual(len(moves), optimum)
                self.assertEqual(self.replay(layout, moves), pp.GOAL)

    def test_manhattan_expands_fewer_nodes(self):
        for layout in pp.COMPARISON_LAYOUTS:
            _, n_mis = pp.solve(layout, pp.GOAL, pp.misplaced)
            _, n_man = pp.solve(layout, pp.GOAL, pp.manhattan)
            self.assertLessEqual(n_man, n_mis)

    def test_unsolvable_layout(self):
        layout = (1, 2, 3, 4, 5, 6, 8, 7, 0)
        self.assertFalse(pp.is_solvable(layout))
        self.assertEqual(pp.solve(layout), (None, 0))
        self.assertIsNone(self.bfs_length(layout))

    def test_bad_layout_is_rejected(self):
        with self.assertRaises(ValueError):
            pp.solve((1, 1, 3, 4, 5, 6, 7, 8, 0))


class AisleGameTests(unittest.TestCase):
    def test_alpha_beta_equals_minimax_with_fewer_nodes(self):
        rng = random.Random(11)
        for _ in range(300):
            aisle = [rng.randint(1, 20) for _ in range(rng.randint(1, 11))]
            value, mm_nodes, ab_nodes = game.compare(aisle)
            self.assertLessEqual(ab_nodes, mm_nodes)

    def test_played_game_reaches_the_minimax_value(self):
        rng = random.Random(5)
        for _ in range(100):
            aisle = [rng.randint(1, 20) for _ in range(rng.randint(1, 9))]
            moves, points = game.play(aisle)
            value, _, _ = game.compare(aisle)
            self.assertEqual(points["A"] - points["B"], value)
            self.assertEqual(points["A"] + points["B"], sum(aisle))
            self.assertEqual(len(moves), len(aisle))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import main
        cls.main = main
        cls.plans, cls.result, cls.workers, cls.items, _ = main.run_pipeline()

    def test_every_order_has_a_tour_through_its_items(self):
        self.assertEqual(len(self.plans), 10)
        for plan in self.plans.values():
            tour = plan.picking.order
            self.assertEqual((tour[0], tour[-1]), ("P1", "D1"))
            wanted = {self.items[i].location_id for i in plan.item_ids}
            self.assertEqual(set(tour[1:-1]), wanted)

    def test_assignments_respect_every_constraint(self):
        self.assertTrue(self.result["success"])
        used = set()
        for plan in self.plans.values():
            worker = self.workers[plan.worker]
            self.assertTrue(worker.available)
            self.assertEqual(worker.zone, plan.zone)
            self.assertIn(plan.slot, self.main.SHIFT_SLOTS)
            self.assertNotIn((plan.worker, plan.slot), used)
            used.add((plan.worker, plan.slot))

    def test_urgent_orders_get_high_priority_and_first_slot(self):
        for plan in self.plans.values():
            self.assertEqual(plan.priority == "HIGH", plan.urgent)
            if plan.urgent:
                self.assertEqual(plan.slot, self.main.SHIFT_SLOTS[0])

    def test_fuzzy_urgency_and_priority_agree(self):
        for plan in self.plans.values():
            self.assertIsNotNone(plan.urgency)
            high = plan.urgent or plan.urgency >= self.main.HIGH_PRIORITY_SCORE
            self.assertEqual(plan.priority == "HIGH", high)

    def test_failed_scans_are_checked(self):
        checked = {i: note for p in self.plans.values() for i, note in p.scan_checks.items()}
        self.assertIn("verify manually", checked["I007"])        # Bayes only
        self.assertIn("0.856", checked["I016"])                  # RFID + barcode

    def test_every_scheduled_order_has_a_delay_risk(self):
        from delay_network import delay_risk
        for plan in self.plans.values():
            self.assertAlmostEqual(plan.delay_risk, delay_risk(**plan.delay_evidence))

    def test_reorder_alerts_follow_stock_levels(self):
        alerts = {a for p in self.plans.values() for a in p.reorder_alerts}
        ordered = {i for p in self.plans.values() for i in p.item_ids}
        expected = {i for i in ordered
                    if self.items[i].quantity <= self.main.REORDER_LEVEL}
        self.assertEqual(alerts, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
