"""
Cheapest fulfilment plan for an order, found with AO* search.

An order is an AND-OR problem:
  - AND: every item in the order is needed.
  - OR:  each item can come from one of several sources, and a source can
         itself need several steps (AND), such as booking a transfer and
         then carrying the item in from dispatch.

AO* keeps a best partial solution graph, expands one unexpanded node on it,
revises the cost estimates upward, and repeats until the root is solved.
With heuristic estimates that never overestimate, it returns the cheapest
plan while leaving branches that cannot win unexpanded.

Syllabus topic: Module 1 - AO* search.

Run on its own:   python picking/ao_star.py
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):          # run directly: make project imports work
    sys.path.insert(0, str(ROOT))


@dataclass
class Connector:
    """One way to achieve a node.

    One child  -> a plain OR option.
    Two or more children -> an AND group: every child is needed.
    `cost` is the extra cost of using this connector itself.
    """
    children: tuple
    cost: float = 0
    label: str = ""


@dataclass
class AndOrGraph:
    connectors: dict = field(default_factory=dict)   # node -> [Connector, ...]
    leaf_cost: dict = field(default_factory=dict)    # leaf -> exact cost

    def add_or(self, node, *options):
        """node is achieved by ANY ONE of the options (each a single child)."""
        self.connectors.setdefault(node, [])
        for child in options:
            self.connectors[node].append(Connector((child,), 0, "or"))

    def add_and(self, node, *children, cost=0):
        """node is achieved by ALL the children together (one AND connector)."""
        self.connectors.setdefault(node, []).append(
            Connector(tuple(children), cost, "and"))

    def add_leaf(self, node, cost):
        self.leaf_cost[node] = cost

    def is_leaf(self, node):
        return node not in self.connectors

    def nodes(self):
        seen = set(self.connectors) | set(self.leaf_cost)
        for options in self.connectors.values():
            for conn in options:
                seen.update(conn.children)
        return seen


@dataclass
class AOStarResult:
    cost: float
    plan: dict               # node -> chosen Connector (for non-leaf nodes)
    expansion_order: list    # nodes in the order AO* expanded them
    total_nodes: int


def ao_star(graph, root, heuristic=None):
    """Return the cheapest solution graph below `root`.

    heuristic: node -> estimate of its cost. Every estimate must be at or
    below the real cost (admissible). Missing estimates count as 0.
    """
    h = heuristic or {}
    estimate = {}            # current cost estimate of each expanded node
    best = {}                # node -> marked (currently cheapest) connector
    solved = set()
    expanded = set()
    expansion_order = []

    parents = {}
    for node, options in graph.connectors.items():
        for conn in options:
            for child in conn.children:
                parents.setdefault(child, set()).add(node)

    def cost_of(node):
        return estimate[node] if node in estimate else h.get(node, 0)

    def revise(node):
        """Recompute a node from its children. Return True if it changed."""
        old = (estimate.get(node), node in solved)
        if graph.is_leaf(node):
            estimate[node] = graph.leaf_cost[node]
            solved.add(node)
        else:
            options = [(conn.cost + sum(cost_of(c) for c in conn.children), i, conn)
                       for i, conn in enumerate(graph.connectors[node])]
            value, _, chosen = min(options, key=lambda o: (o[0], o[1]))
            estimate[node] = value
            best[node] = chosen
            if all(c in solved for c in chosen.children):
                solved.add(node)
        return old != (estimate[node], node in solved)

    def next_unexpanded(node):
        """Follow the marked connectors down to an unexpanded node."""
        if node not in expanded:
            return node
        for child in best[node].children:
            if child not in solved:
                found = next_unexpanded(child)
                if found is not None:
                    return found
        return None

    while root not in solved:
        node = next_unexpanded(root)
        if node is None:                       # cannot happen on a valid graph
            raise RuntimeError("AO* found no node to expand")

        expanded.add(node)
        expansion_order.append(node)

        # Revise this node, then every ancestor whose estimate may change.
        work = [node]
        while work:
            current = work.pop()
            if revise(current):
                work.extend(p for p in parents.get(current, ()) if p in expanded)

    plan = {}

    def collect(node):
        if not graph.is_leaf(node):
            plan[node] = best[node]
            for child in best[node].children:
                collect(child)

    collect(root)
    return AOStarResult(estimate[root], plan, expansion_order, len(graph.nodes()))


def exhaustive_cost(graph, node):
    """Cheapest cost by checking every option. Used to verify AO*."""
    if graph.is_leaf(node):
        return graph.leaf_cost[node]
    return min(conn.cost + sum(exhaustive_cost(graph, c) for c in conn.children)
               for conn in graph.connectors[node])


def lower_bounds(graph, leaf_bounds):
    """Admissible estimates for every node from estimates for the leaves.

    An OR node can cost no less than its cheapest option, and an AND group
    no less than the sum of its children, so the bounds stay admissible.
    """
    bounds = {}

    def bound(node):
        if node not in bounds:
            if graph.is_leaf(node):
                bounds[node] = leaf_bounds.get(node, 0)
            else:
                bounds[node] = min(conn.cost + sum(bound(c) for c in conn.children)
                                   for conn in graph.connectors[node])
        return bounds[node]

    for node in graph.nodes():
        bound(node)
    return bounds


def print_plan(graph, result, root):
    def show(node, depth):
        pad = "    " * depth
        if graph.is_leaf(node):
            print(f"{pad}- {node}  ({graph.leaf_cost[node]:.0f} m)")
            return
        conn = result.plan[node]
        kind = "ALL of" if len(conn.children) > 1 else "chosen option"
        print(f"{pad}{node}  [{kind}]")
        for child in conn.children:
            show(child, depth + 1)

    show(root, 0)


# ---------------------------------------------------------------------------
# Demo: order ORD201, a workstation kit.
#
# Primary shelf positions come from data/inventory.csv. The returns bay,
# the partner-DC transfer and its 40 m booking cost are example scenario
# data for this demo only.
#
# Leaf cost = round trip in metres from the picking station P1, using A*
# on the warehouse graph. Leaf estimates = round trip x the admissible
# Manhattan heuristic from picking/route_costs.py.
# ---------------------------------------------------------------------------

TRANSFER_BOOKING_COST = 40


def build_demo(router):
    def trip(location):
        return 2 * router.distance("P1", location)

    def trip_bound(location):
        loc_a, loc_b = router.graph.get_location("P1"), router.graph.get_location(location)
        return 2 * router.scale * (abs(loc_a.x - loc_b.x) + abs(loc_a.y - loc_b.y))

    g = AndOrGraph()
    g.add_and("ORD201", "Keyboard", "Monitor", "Packing material")

    g.add_or("Keyboard", "Keyboard from shelf S1", "Keyboard from returns bay L08")
    g.add_leaf("Keyboard from shelf S1", trip("S1"))
    g.add_leaf("Keyboard from returns bay L08", trip("L08"))

    g.add_or("Monitor", "Monitor from shelf S3", "Monitor by partner-DC transfer")
    g.add_leaf("Monitor from shelf S3", trip("S3"))
    g.add_and("Monitor by partner-DC transfer",
              "Book transfer slot", "Carry from dispatch D1")
    g.add_leaf("Book transfer slot", TRANSFER_BOOKING_COST)
    g.add_leaf("Carry from dispatch D1", 2 * router.distance("D1", "P1"))

    g.add_or("Packing material", "Bubble wrap from L05", "Tape and labels")
    g.add_leaf("Bubble wrap from L05", trip("L05"))
    g.add_and("Tape and labels", "Packing tape from L04", "Label roll from L10")
    g.add_leaf("Packing tape from L04", trip("L04"))
    g.add_leaf("Label roll from L10", trip("L10"))

    leaf_bounds = {
        "Keyboard from shelf S1": trip_bound("S1"),
        "Keyboard from returns bay L08": trip_bound("L08"),
        "Monitor from shelf S3": trip_bound("S3"),
        "Book transfer slot": TRANSFER_BOOKING_COST,
        "Carry from dispatch D1": 2 * router.scale * 20,   # D1-P1 Manhattan = 20 m
        "Bubble wrap from L05": trip_bound("L05"),
        "Packing tape from L04": trip_bound("L04"),
        "Label roll from L10": trip_bound("L10"),
    }
    return g, lower_bounds(g, leaf_bounds)


def run_demo():
    from picking.route_costs import GraphRouter

    router = GraphRouter.from_file()
    graph, bounds = build_demo(router)
    result = ao_star(graph, "ORD201", bounds)
    blind = ao_star(graph, "ORD201")                    # every estimate = 0

    print("ORD201 needs a keyboard, a monitor and packing material.")
    print("Each can come from more than one source (OR); some sources need")
    print("several steps (AND). Costs are round trips in metres from P1.")
    print()
    print("Cheapest fulfilment plan (AO*):")
    print_plan(graph, result, "ORD201")
    print()
    print(f"Total cost: {result.cost:.0f} m "
          f"(checked against every option: {exhaustive_cost(graph, 'ORD201'):.0f} m)")
    print()
    print("Expansion order:")
    for step, node in enumerate(result.expansion_order, 1):
        print(f"  {step:>2}. {node}")
    never = sorted(graph.nodes() - set(result.expansion_order))
    print()
    print(f"Nodes expanded: {len(result.expansion_order)} of {result.total_nodes}")
    print("Never expanded, because their estimate already showed they cannot win:")
    for node in never:
        print(f"  - {node}")
    print()
    print(f"Same search with every estimate set to 0: "
          f"{len(blind.expansion_order)} nodes expanded, cost {blind.cost:.0f} m")
    return result


if __name__ == "__main__":
    run_demo()
