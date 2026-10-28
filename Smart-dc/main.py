"""
Smart Distribution Center Management System
Main Program

Single destination:
    P1 -> Goal
    A* shortest path

Multiple destinations:
    P1 -> item locations -> D1
    TSP decides visiting order
    A* finds the physical shortest path for every leg
"""

from routing.warehouse import (
    location_to_cell,
    print_warehouse,
    WAREHOUSE_GRID
)

from routing.bfs import bfs

from routing.astar import (
    astar,
    astar_locations,
    astar_distance
)

from picking.tsp_exact import (
    exact_picking_order,
    format_plan
)

from picking.tsp_nearest import (
    nearest_neighbour_order
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_separator():
    print("=" * 70)


def print_route(path):
    """Print grid-cell path in readable form."""

    if not path:
        print("No route found.")
        return

    print(
        " -> ".join(
            str(cell)
            for cell in path
        )
    )


def calculate_node_reduction(
    bfs_nodes,
    astar_nodes
):
    """Calculate percentage reduction in explored nodes."""

    if bfs_nodes == 0:
        return 0.0

    return (
        (bfs_nodes - astar_nodes)
        / bfs_nodes
    ) * 100


def calculate_extra_distance(
    exact_cost,
    nearest_cost
):
    """Calculate percentage extra distance."""

    if exact_cost == 0:
        return 0.0

    return (
        (nearest_cost - exact_cost)
        / exact_cost
    ) * 100


# ============================================================
# BFS VS A* COMPARISON
# ============================================================

def run_route_comparison(
    start_location,
    goal_location
):
    """Compare BFS and A* for one route."""

    print()
    print_separator()
    print("ROUTE ALGORITHM COMPARISON")
    print_separator()

    print(
        f"Start Location : {start_location}"
    )

    print(
        f"Goal Location  : {goal_location}"
    )

    try:
        start = location_to_cell(
            start_location
        )

        goal = location_to_cell(
            goal_location
        )

    except Exception as error:

        print()
        print(
            f"Invalid location: {error}"
        )

        return

    # --------------------------------------------------------
    # BFS
    # --------------------------------------------------------

    print()
    print("Running BFS...")

    bfs_result = bfs(
        WAREHOUSE_GRID,
        start,
        goal
    )

    # --------------------------------------------------------
    # A*
    # --------------------------------------------------------

    print("Running A*...")

    astar_result = astar(
        WAREHOUSE_GRID,
        start,
        goal
    )

    # --------------------------------------------------------
    # BFS RESULT
    # --------------------------------------------------------

    print()
    print_separator()
    print("BFS RESULT")
    print_separator()

    if bfs_result["path"] is None:

        print("No BFS route found.")

    else:

        print(
            f"Location Route : "
            f"{start_location} -> {goal_location}"
        )

        print(
            "Physical Path   : ",
            end=""
        )

        print_route(
            bfs_result["path"]
        )

        print(
            f"Cost            : "
            f"{bfs_result['cost']} m"
        )

        print(
            f"Nodes Explored  : "
            f"{bfs_result['nodes_explored']}"
        )

    # --------------------------------------------------------
    # A* RESULT
    # --------------------------------------------------------

    print()
    print_separator()
    print("A* RESULT")
    print_separator()

    if astar_result["path"] is None:

        print("No A* route found.")

    else:

        print(
            f"Location Route : "
            f"{start_location} -> {goal_location}"
        )

        print(
            "Physical Path   : ",
            end=""
        )

        print_route(
            astar_result["path"]
        )

        print(
            f"Cost            : "
            f"{astar_result['cost']} m"
        )

        print(
            f"Nodes Explored  : "
            f"{astar_result['nodes_expanded']}"
        )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    if (
        bfs_result["path"] is not None
        and
        astar_result["path"] is not None
    ):

        reduction = calculate_node_reduction(
            bfs_result["nodes_explored"],
            astar_result["nodes_expanded"]
        )

        print()
        print_separator()
        print("COMPARISON")
        print_separator()

        print(
            f"BFS Distance   : "
            f"{bfs_result['cost']} m"
        )

        print(
            f"A* Distance    : "
            f"{astar_result['cost']} m"
        )

        print(
            f"BFS Nodes      : "
            f"{bfs_result['nodes_explored']}"
        )

        print(
            f"A* Nodes       : "
            f"{astar_result['nodes_expanded']}"
        )

        print(
            f"A* Node Reduction : "
            f"{reduction:.2f}%"
        )


# ============================================================
# A* SINGLE LEG
# ============================================================

def calculate_a_star_leg(
    start_location,
    goal_location
):
    """
    Calculate one A* leg.

    Displays:
        Location route
        Physical grid path
        Distance
        Nodes explored
    """

    result = astar_locations(
        start_location,
        goal_location
    )

    print()
    print_separator()

    print(
        f"A* LEG: {start_location} -> {goal_location}"
    )

    print_separator()

    if result["path"] is None:

        print("No route found.")

        return None

    print(
        f"Location Route : "
        f"{start_location} -> {goal_location}"
    )

    print(
        "Physical Path  : ",
        end=""
    )

    print_route(
        result["path"]
    )

    print(
        f"Distance       : "
        f"{result['cost']} m"
    )

    print(
        f"Nodes Explored : "
        f"{result['nodes_expanded']}"
    )

    return result


# ============================================================
# TSP + A* ROUTES
# ============================================================

def print_tsp_legs(
    visit_order
):
    """
    Run A* between every consecutive location
    in the TSP visiting order.
    """

    print()
    print_separator()
    print("A* ROUTE FOR EACH TSP LEG")
    print_separator()

    total_astar_cost = 0

    for index in range(
        len(visit_order) - 1
    ):

        start_location = visit_order[index]

        goal_location = visit_order[
            index + 1
        ]

        result = calculate_a_star_leg(
            start_location,
            goal_location
        )

        if result is None:
            return None

        total_astar_cost += result["cost"]

    print()
    print_separator()
    print("COMPLETE A* ROUTE")
    print_separator()

    print(
        "Location Route : "
        +
        " -> ".join(
            visit_order
        )
    )

    print(
        f"Total Distance : "
        f"{total_astar_cost} m"
    )

    return total_astar_cost


# ============================================================
# MULTI-LOCATION ORDER
# ============================================================

def run_optimized_picking(
    order_id,
    locations
):
    """
    Multi-location picking:

        Exact TSP
            ↓
        Optimal visit order
            ↓
        A* for every leg
    """

    print()
    print_separator()
    print("MULTI-LOCATION ORDER")
    print_separator()

    print(
        f"Order ID : {order_id}"
    )

    print(
        "Required Locations : "
        +
        ", ".join(locations)
    )

    # --------------------------------------------------------
    # EXACT TSP
    # --------------------------------------------------------

    print()
    print(
        "Calculating exact TSP order using A* distances..."
    )

    exact_plan = exact_picking_order(
        locations,
        astar_distance
    )

    print()
    print_separator()
    print("EXACT TSP RESULT")
    print_separator()

    print(
        format_plan(
            exact_plan
        )
    )

    # --------------------------------------------------------
    # NEAREST NEIGHBOUR
    # --------------------------------------------------------

    print()
    print_separator()
    print("NEAREST-NEIGHBOUR RESULT")
    print_separator()

    nearest_plan = nearest_neighbour_order(
        locations,
        astar_distance
    )

    print(
        format_plan(
            nearest_plan
        )
    )

    # --------------------------------------------------------
    # TSP COMPARISON
    # --------------------------------------------------------

    extra_distance = calculate_extra_distance(
        exact_plan.cost,
        nearest_plan.cost
    )

    print()
    print_separator()
    print("TSP COMPARISON")
    print_separator()

    print(
        f"Exact TSP Cost       : "
        f"{exact_plan.cost} m"
    )

    print(
        f"Nearest Neighbour    : "
        f"{nearest_plan.cost} m"
    )

    print(
        f"Extra Distance       : "
        f"{extra_distance:.2f}%"
    )

    # --------------------------------------------------------
    # OPTIMAL ORDER
    # --------------------------------------------------------

    print()
    print_separator()
    print("OPTIMAL PICKING ORDER")
    print_separator()

    print(
        " -> ".join(
            exact_plan.order
        )
    )

    # --------------------------------------------------------
    # A* FOR EVERY LEG
    # --------------------------------------------------------

    total_astar_cost = print_tsp_legs(
        exact_plan.order
    )

    # --------------------------------------------------------
    # FINAL PLAN
    # --------------------------------------------------------

    print()
    print_separator()
    print("FINAL OPTIMIZED PLAN")
    print_separator()

    print(
        f"Order ID       : "
        f"{order_id}"
    )

    print(
        f"Picking Order  : "
        f"{' -> '.join(exact_plan.order)}"
    )

    if total_astar_cost is not None:

        print(
            f"Total Distance : "
            f"{total_astar_cost} m"
        )

    print(
        "Algorithm      : "
        "Exact TSP + A*"
    )

    print(
        "Status         : "
        "READY FOR PICKING"
    )

    print_separator()


# ============================================================
# SINGLE GOAL LIVE SIMULATION
# ============================================================

def run_single_goal_simulation():
    """
    Interactive single-goal simulation.

    Start location is P1.
    """

    print()
    print_separator()
    print("SMART DISTRIBUTION CENTER - LIVE SIMULATION")
    print_separator()

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    order_id = input(
        "Enter Order ID: "
    ).strip()

    goal_location = input(
        "Enter Goal Location: "
    ).strip().upper()

    priority = input(
        "Enter Priority (LOW/MEDIUM/HIGH): "
    ).strip().upper()

    start_location = "P1"

    # --------------------------------------------------------
    # PRIORITY CHECK
    # --------------------------------------------------------

    valid_priorities = [
        "LOW",
        "MEDIUM",
        "HIGH"
    ]

    if priority not in valid_priorities:

        print()
        print(
            "Invalid priority."
        )

        print(
            "Please enter LOW, MEDIUM or HIGH."
        )

        return

    # --------------------------------------------------------
    # ORDER RECEIVED
    # --------------------------------------------------------

    print()
    print_separator()
    print("ORDER RECEIVED")
    print_separator()

    print(
        f"Order ID        : "
        f"{order_id}"
    )

    print(
        f"Priority        : "
        f"{priority}"
    )

    print(
        f"Start Location  : "
        f"{start_location}"
    )

    print(
        f"Goal Location   : "
        f"{goal_location}"
    )

    # --------------------------------------------------------
    # A*
    # --------------------------------------------------------

    print()
    print(
        "Calculating shortest route using A*..."
    )

    try:

        result = astar_locations(
            start_location,
            goal_location
        )

    except Exception as error:

        print()
        print(
            f"Could not calculate route: {error}"
        )

        return

    # --------------------------------------------------------
    # NO ROUTE
    # --------------------------------------------------------

    if result["path"] is None:

        print()
        print_separator()
        print("ROUTE NOT FOUND")
        print_separator()

        print(
            f"No route exists from "
            f"{start_location} "
            f"to "
            f"{goal_location}."
        )

        return

    # --------------------------------------------------------
    # A* RESULT
    # --------------------------------------------------------

    print()
    print_separator()
    print("SELECTED A* ROUTE")
    print_separator()

    print(
        f"Location Route  : "
        f"{start_location} -> {goal_location}"
    )

    print(
        "Physical Path   : ",
        end=""
    )

    print_route(
        result["path"]
    )

    print(
        f"Distance        : "
        f"{result['cost']} m"
    )

    print(
        f"Nodes Explored  : "
        f"{result['nodes_expanded']}"
    )

    # --------------------------------------------------------
    # FINAL PLAN
    # --------------------------------------------------------

    print()
    print_separator()
    print("FINAL PLAN")
    print_separator()

    print(
        f"Order           : "
        f"{order_id}"
    )

    print(
        f"Priority        : "
        f"{priority}"
    )

    print(
        f"Start           : "
        f"{start_location}"
    )

    print(
        f"Goal            : "
        f"{goal_location}"
    )

    print(
        f"Route           : "
        f"{start_location} -> {goal_location}"
    )

    print(
        "Physical Path   : ",
        end=""
    )

    print_route(
        result["path"]
    )

    print(
        f"Distance        : "
        f"{result['cost']} m"
    )

    print(
        "Algorithm       : "
        "A* Search"
    )

    print(
        "Status          : "
        "READY FOR PICKING"
    )

    print_separator()


# ============================================================
# TSP DEMONSTRATIONS
# ============================================================

def run_tsp_demo():
    """Run sample TSP orders."""

    print()
    print_separator()
    print("TSP DEMONSTRATIONS")
    print_separator()

    demo_orders = {

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

    for order_id, locations in demo_orders.items():

        run_optimized_picking(
            order_id,
            locations
        )

        print()


# ============================================================
# MAIN MENU
# ============================================================

def main():

    while True:

        print()
        print_separator()
        print(
            "SMART DISTRIBUTION CENTER MANAGEMENT SYSTEM"
        )
        print_separator()

        print(
            "1. Single Goal Route - A*"
        )

        print(
            "2. Multi-Location Order - TSP + A*"
        )

        print(
            "3. BFS vs A* Comparison"
        )

        print(
            "4. TSP Demonstrations"
        )

        print(
            "5. Show Warehouse"
        )

        print(
            "6. Exit"
        )

        print_separator()

        choice = input(
            "Enter choice: "
        ).strip()

        # ----------------------------------------------------
        # OPTION 1
        # ----------------------------------------------------

        if choice == "1":

            run_single_goal_simulation()

        # ----------------------------------------------------
        # OPTION 2
        # ----------------------------------------------------

        elif choice == "2":

            print()

            order_id = input(
                "Enter Order ID: "
            ).strip()

            location_input = input(
                "Enter warehouse locations "
                "(comma separated): "
            )

            locations = [
                location.strip().upper()
                for location in
                location_input.split(",")
                if location.strip()
            ]

            if not locations:

                print(
                    "No locations entered."
                )

                continue

            run_optimized_picking(
                order_id,
                locations
            )

        # ----------------------------------------------------
        # OPTION 3
        # ----------------------------------------------------

        elif choice == "3":

            print()

            start_location = input(
                "Enter Start Location: "
            ).strip().upper()

            goal_location = input(
                "Enter Goal Location: "
            ).strip().upper()

            run_route_comparison(
                start_location,
                goal_location
            )

        # ----------------------------------------------------
        # OPTION 4
        # ----------------------------------------------------

        elif choice == "4":

            run_tsp_demo()

        # ----------------------------------------------------
        # OPTION 5
        # ----------------------------------------------------

        elif choice == "5":

            print()
            print_separator()
            print("WAREHOUSE MAP")
            print_separator()

            print_warehouse()

        # ----------------------------------------------------
        # OPTION 6
        # ----------------------------------------------------

        elif choice == "6":

            print()
            print(
                "Exiting Smart Distribution Center..."
            )

            break

        # ----------------------------------------------------
        # INVALID
        # ----------------------------------------------------

        else:

            print()
            print(
                "Invalid choice."
            )

            print(
                "Please select 1-6."
            )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()