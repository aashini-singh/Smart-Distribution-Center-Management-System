"""
A* route costs on the warehouse graph (data/warehouse.txt).

Why this file exists
--------------------
routing/astar.py searches the 6 x 10 grid, which contains only P1, D1 and
S1-S9. Items I011-I020 are stored at L01-L10, which exist only in the
warehouse graph loaded by warehouse/graph.py. The full pipeline in main.py
plans real orders for all 20 items, so it needs A* on that graph.

Heuristic
---------
h(n) = scale x Manhattan distance from n to the goal.

Plain Manhattan distance overestimates on one aisle: S5-L01 is 10 m long,
but the two locations are 15 m apart in Manhattan distance. So plain
Manhattan is not admissible here. With today's layout it still happens to
find the shortest route for every pair, but A* is only guaranteed optimal
with an admissible heuristic, and a later layout change could break it.
The scale is the smallest (edge cost / Manhattan length) over all edges,
which is 2/3 for the current data, so h(n) never overestimates.

Syllabus topic: Module 1 - Heuristic, A* search.

Run on its own:   python picking/route_costs.py
"""

import heapq
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):          # run directly: make project imports work
    sys.path.insert(0, str(ROOT))

from warehouse.graph import load_warehouse

DEFAULT_LAYOUT = ROOT / "data" / "warehouse.txt"


def manhattan(graph, a, b):
    """Manhattan distance in metres between two locations of the graph."""
    la, lb = graph.get_location(a), graph.get_location(b)
    return abs(la.x - lb.x) + abs(la.y - lb.y)


def heuristic_scale(graph):
    """Largest scale that keeps scale x Manhattan at or below every edge cost."""
    scale = 1.0
    for a, neighbours in graph.edges.items():
        for b, cost in neighbours:
            length = manhattan(graph, a, b)
            if length > 0:
                scale = min(scale, cost / length)
    return scale


def graph_astar(graph, start, goal, scale=None):
    """Least-cost route between two locations of the warehouse graph.

    Returns a dictionary with path, cost (metres) and nodes_expanded,
    the same shape as routing/astar.py returns.
    """
    for location in (start, goal):
        if graph.get_location(location) is None:
            raise ValueError(f"Unknown warehouse location: {location}")

    if scale is None:
        scale = heuristic_scale(graph)

    def h(node):
        return scale * manhattan(graph, node, goal)

    open_list = [(h(start), 0, start)]
    g_cost = {start: 0}
    parent = {start: None}
    nodes_expanded = 0

    while open_list:
        _, current_g, current = heapq.heappop(open_list)

        if current_g != g_cost[current]:      # outdated queue entry
            continue

        nodes_expanded += 1

        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return {"path": path, "cost": current_g,
                    "nodes_expanded": nodes_expanded}

        for neighbour, step_cost in graph.get_neighbors(current):
            new_g = current_g + step_cost
            if neighbour not in g_cost or new_g < g_cost[neighbour]:
                g_cost[neighbour] = new_g
                parent[neighbour] = current
                heapq.heappush(open_list, (new_g + h(neighbour), new_g, neighbour))

    return {"path": None, "cost": None, "nodes_expanded": nodes_expanded}


class GraphRouter:
    """A* on the warehouse graph, with each route computed only once.

    router.distance is the function the TSP code expects:
        exact_picking_order(["S1", "L05"], router.distance)
    """

    def __init__(self, graph):
        self.graph = graph
        self.scale = heuristic_scale(graph)
        self._routes = {}

    @classmethod
    def from_file(cls, filename=DEFAULT_LAYOUT):
        return cls(load_warehouse(filename))

    def route(self, start, goal):
        key = (start, goal)
        if key not in self._routes:
            self._routes[key] = graph_astar(self.graph, start, goal, self.scale)
        return self._routes[key]

    def distance(self, start, goal):
        cost = self.route(start, goal)["cost"]
        return float("inf") if cost is None else cost


if __name__ == "__main__":
    router = GraphRouter.from_file()
    print(f"Heuristic scale: {router.scale:.3f} "
          f"(h = {router.scale:.3f} x Manhattan distance)")
    print()
    for start, goal in [("P1", "S9"), ("P1", "L05"), ("S1", "L21"), ("D1", "L10")]:
        result = router.route(start, goal)
        print(f"{start} -> {goal}: {' -> '.join(result['path'])}")
        print(f"{'':<10}cost = {result['cost']} m, "
              f"nodes expanded = {result['nodes_expanded']}")
