"""
Smart Distribution Center Management System
--------------------------------------------

Main demonstration program.

Current implementation:

Module 1:
    - Breadth-First Search (BFS)
    - A* Search
    - Manhattan Distance Heuristic
    - Route cost comparison
    - Search effort comparison

Module 2:
    - Exact TSP
    - Nearest Neighbour heuristic
    - A* route cost integration

The routing module determines the physical route between
two warehouse locations.

The picking module determines the order in which multiple
warehouse locations should be visited.
"""


from routing.warehouse import (
    location_to_cell,
    print_warehouse,
    WAREHOUSE_GRID
)

from routing.bfs import bfs

from routing.astar import (
    astar,
    astar_distance
)

from picking.tsp_exact import (
    exact_picking_order,
    format_plan
)

from picking.tsp_nearest import (
    nearest_neighbour_order
)
from rules.explain import why
from rules.fol_rules import load_kb, forward_chain, RULES as FOL_RULES
from rules.certainty import reorder_advice, label as certainty_label
from rules.fuzzy_priority import urgency, label as urgency_label

# ============================================================
# Utility Functions
# ============================================================

def print_route(result):
    """
    Print route information returned by BFS or A*.
    """

    if result["path"] is None:
        print("No route found.")
        return

    print("Route:")

    print(
        " -> ".join(
            str(cell)
            for cell in result["path"]
        )
    )

    print(
        f"Path cost       : "
        f"{result['cost']} m"
    )

    print(
        f"Nodes expanded  : "
        f"{result['nodes_expanded']}"
    )


def calculate_node_reduction(
    bfs_nodes,
    astar_nodes
):
    """
    Calculate the percentage reduction in nodes expanded
    by A* compared with BFS.
    """

    if bfs_nodes == 0:
        return 0.0

    reduction = (
        (
            bfs_nodes
            -
            astar_nodes
        )
        /
        bfs_nodes
    ) * 100

    return reduction


def calculate_extra_distance(
    exact_cost,
    nearest_cost
):
    """
    Calculate how much farther the nearest-neighbour
    solution travels compared with the exact TSP solution.
    """

    if exact_cost == 0:
        return 0.0

    extra = (
        (
            nearest_cost
            -
            exact_cost
        )
        /
        exact_cost
    ) * 100

    return extra


# ============================================================
# BFS vs A* Route Comparison
# ============================================================

def run_route_comparison(
    start_location,
    goal_location
):
    """
    Compare BFS and A* for a pair of warehouse locations.

    Metrics:
        - route
        - path cost
        - nodes expanded
        - A* node-expansion reduction
    """

    start = location_to_cell(
        start_location
    )

    goal = location_to_cell(
        goal_location
    )

    print()
    print("=" * 60)
    print("ROUTE SEARCH COMPARISON")
    print("=" * 60)

    print(
        f"Start: {start_location} "
        f"{start}"
    )

    print(
        f"Goal : {goal_location} "
        f"{goal}"
    )

    # --------------------------------------------------------
    # BFS
    # --------------------------------------------------------

    print()
    print("--------------- BFS ----------------")

    bfs_result = bfs(
        WAREHOUSE_GRID,
        start,
        goal
    )

    print_route(
        bfs_result
    )

    # --------------------------------------------------------
    # A*
    # --------------------------------------------------------

    print()
    print("--------------- A* -----------------")

    astar_result = astar(
        WAREHOUSE_GRID,
        start,
        goal
    )

    print_route(
        astar_result
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print()
    print("------------- COMPARISON ------------")

    print(
        f"{'Metric':<25}"
        f"{'BFS':<15}"
        f"{'A*':<15}"
    )

    print("-" * 55)

    print(
        f"{'Path cost (m)':<25}"
        f"{str(bfs_result['cost']):<15}"
        f"{str(astar_result['cost']):<15}"
    )

    print(
        f"{'Nodes expanded':<25}"
        f"{str(bfs_result['nodes_expanded']):<15}"
        f"{str(astar_result['nodes_expanded']):<15}"
    )

    # --------------------------------------------------------
    # Node expansion improvement
    # --------------------------------------------------------

    reduction = calculate_node_reduction(
        bfs_result["nodes_expanded"],
        astar_result["nodes_expanded"]
    )

    print()

    print(
        f"A* node-expansion reduction: "
        f"{reduction:.2f}%"
    )

    # --------------------------------------------------------
    # Path cost comparison
    # --------------------------------------------------------

    if (
        bfs_result["cost"] is not None
        and
        astar_result["cost"] is not None
    ):

        if bfs_result["cost"] == astar_result["cost"]:

            print(
                "Result: BFS and A* found "
                "routes with the same path cost."
            )

        elif astar_result["cost"] < bfs_result["cost"]:

            print(
                "Result: A* found a lower-cost route."
            )

        else:

            print(
                "Result: BFS found a lower-cost route."
            )

    # --------------------------------------------------------
    # A* warehouse visualization
    # --------------------------------------------------------

    print()
    print("Warehouse with A* route:")

    print_warehouse(
        path=astar_result["path"],
        start=start,
        goal=goal
    )

    return bfs_result, astar_result


# ============================================================
# TSP Picking Order Comparison
# ============================================================

def run_tsp_demo():
    """
    Demonstrate the existing TSP module.

    The TSP uses A* route cost instead of direct
    Manhattan distance.

    This connects:

        TSP
          ↓
        A*
          ↓
    Actual warehouse route cost
    """

    print()
    print("=" * 60)
    print("PICKING ORDER OPTIMIZATION")
    print("=" * 60)

    orders = {

        "ORD101": [
            "S2",
            "S4",
            "S7"
        ],

        "ORD102": [
            "S1",
            "S3",
            "S4",
            "S6",
            "S9"
        ],

        "ORD103": [
            "S5",
            "S5",
            "S8"
        ]
    }

    for order_id, locations in orders.items():

        print()
        print(
            f"{order_id}: {locations}"
        )

        # ----------------------------------------------------
        # Exact TSP
        # ----------------------------------------------------

        exact_plan = exact_picking_order(
            locations,
            astar_distance
        )

        print()
        print(
            format_plan(
                exact_plan
            )
        )

        # ----------------------------------------------------
        # Nearest Neighbour
        # ----------------------------------------------------

        nearest_plan = nearest_neighbour_order(
            locations,
            astar_distance
        )

        print(
            format_plan(
                nearest_plan
            )
        )

        # ----------------------------------------------------
        # Comparison
        # ----------------------------------------------------

        extra_distance = calculate_extra_distance(
            exact_plan.cost,
            nearest_plan.cost
        )

        print()

        if extra_distance > 0:

            print(
                f"Nearest-neighbour extra distance: "
                f"{extra_distance:.2f}%"
            )

        elif extra_distance == 0:

            print(
                "Nearest-neighbour found the "
                "same-cost route as the exact method."
            )

        else:

            print(
                f"Nearest-neighbour is "
                f"{abs(extra_distance):.2f}% "
                f"shorter than the exact result."
            )


def run_expert_system_demo():
    print("\n" + "=" * 60)
    print("RULE-BASED EXPERT SYSTEM")
    print("=" * 60)

    # 1. IF-THEN rules and explanations
    facts = {
        "order_id": "ORD101",
        "urgent": True,
        "stock": 3,
        "required_quantity": 5,
        "reorder_level": 4,
    }

    final_facts, explanations = why(facts)

    print("\nOrder decisions:")
    for key, value in final_facts.items():
        print(f"  {key}: {value}")

    print("\nWhy the rules fired:")
    for explanation in explanations:
        print(" ", explanation)

    # 2. First-order logic
    print("\nFirst-order logic:")
    kb = load_kb()
    derived = forward_chain(kb, FOL_RULES)
    print(f"  Initial knowledge base: {len(kb.facts) - len(derived)} facts")
    print(f"  New conclusions: {len(derived)}")

    for fact in sorted(derived):
        if fact[0] in ("Priority", "ReorderAlert", "NeedsProtection", "PackFirst"):
            print(" ", fact)

    # 3. Certainty factors
    print("\nCertainty-factor assessment:")
    reorder_cf, delay_cf, _ = reorder_advice(
        stock=3,
        reorder_level=4,
        demand_rising=0.7,
        supplier_late=0.8,
    )
    print(f"  Reorder confidence: {reorder_cf:.2f}")
    print(f"  Supplier-delay confidence: {delay_cf:.2f}")
    print(f"  Recommendation: {certainty_label(reorder_cf)}")

    # 4. Fuzzy urgency
    print("\nFuzzy urgency:")
    score, _ = urgency(hours=5, value=80)
    print(f"  Urgency score: {score:.1f}/100")
    print(f"  Urgency level: {urgency_label(score)}")

# ============================================================
# Main Program
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "SMART DISTRIBUTION CENTER "
        "MANAGEMENT SYSTEM"
    )
    print("=" * 60)

    print()
    print("Current AI implementation:")

    print(
        "1. BFS baseline routing"
    )

    print(
        "2. A* heuristic routing"
    )

    print(
        "3. Manhattan distance heuristic"
    )

    print(
        "4. TSP integration using A* route cost"
    )

    # --------------------------------------------------------
    # Route search demonstration
    # --------------------------------------------------------
    #
    # P1 = picking station
    # S9 = storage location
    #
    # This compares BFS and A* on the same warehouse.
    # --------------------------------------------------------

    run_route_comparison(
        "P1",
        "S9"
    )

    # --------------------------------------------------------
    # TSP demonstration
    # --------------------------------------------------------

    run_tsp_demo()

    run_tsp_demo()
    
    run_expert_system_demo()
    # --------------------------------------------------------
    # End
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PROGRAM COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print()


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":
    main()