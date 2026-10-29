"""
Smart Distribution Center Management System
--------------------------------------------

Main program.

    python main.py              plan all 10 test orders end to end
    python main.py ORD104       full plan for one order
    python main.py demos        run every algorithm demo
    python main.py demo NAME    run one demo: routing, picking, ao, csp, pallet,
                                game, queens, lift, fol, explain, certainty,
                                fuzzy, bayesnet

Order pipeline (one order, start to finish):

    1. Rules (Abhirag)        fuzzy urgency, expert system (priority, stock hold,
                              reorder alert), certainty factor on each alert
    2. Item check (Aashini)   item -> shelf; a failed scan is checked with Bayes,
                              or Dempster-Shafer when RFID and barcode both read
    3. Route costs            A* on the warehouse graph (picking/route_costs.py)
    4. Picking order          TSP: exact up to 8 stops, nearest neighbour above
    5. Worker + slot (Ananya) CSP with MRV and forward checking
    6. Delay risk (Ananya)    Bayesian network over congestion, shortage, stock-out

Algorithm demos: BFS vs A* and TSP on the grid (Amulya), AO* fulfilment plan,
pallet puzzle (tiles problem, A*), and aisle game (minimax, alpha-beta).
"""

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

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


from picking.tsp_nearest import plan_picking
from picking.route_costs import GraphRouter

from rules import RuleEngine, RULES
from rules.fuzzy_priority import urgency as fuzzy_urgency, HIGH_PRIORITY_SCORE
from rules.certainty import reorder_advice, label as reorder_label

from uncertainty.bayes_scan import assess_scan
from uncertainty.dempster_shafer import combine_evidence

from warehouse.records import load_items, load_workers

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# scheduling/solver.py imports "from model import ...", so its folder must be
# on the import path.
SCHEDULING_DIR = ROOT / "scheduling"
if str(SCHEDULING_DIR) not in sys.path:
    sys.path.append(str(SCHEDULING_DIR))

import model as scheduling_model      # scheduling/model.py  (Ananya)
import solver as scheduling_solver    # scheduling/solver.py (Ananya)
from solver_mrv import find_schedule_mrv            # scheduling/solver_mrv.py
from delay_network import delay_risk, risk_label    # scheduling/delay_network.py


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


# ============================================================
# Order Pipeline
# ============================================================
#
# Two values the data files do not hold yet. Both are used only here and
# can move into the CSV files later:
#
#   REORDER_LEVEL  inventory.csv has no reorder-level column.
#   SHIFT_SLOTS    workers.csv has no time slots, so every available
#                  worker gets the same four one-hour slots.
#
# Delay-risk evidence (step 6), read from the finished schedule:
#   congestion  the order's slot holds BUSY_SLOT_ORDERS or more orders
#   shortage    the order's zone has only one available worker
#   stock-out   a line of the order is on stock hold
# ------------------------------------------------------------

REORDER_LEVEL = 10
SHIFT_SLOTS = ["09-10", "10-11", "11-12", "12-13"]
BUSY_SLOT_ORDERS = 4


@dataclass
class OrderPlan:
    order_id: str
    zone: str
    urgent: bool
    item_ids: list
    deadline_hours: float = None
    order_value: float = None
    urgency: float = None           # fuzzy score 0-100
    priority: str = "NORMAL"
    fired_rules: list = field(default_factory=list)
    stock_holds: list = field(default_factory=list)
    reorder_alerts: list = field(default_factory=list)
    reorder_cf: dict = field(default_factory=dict)      # item -> certainty factor
    scan_checks: dict = field(default_factory=dict)     # item -> note
    picking: object = None          # PickingPlan from picking/tsp_exact.py
    legs: list = field(default_factory=list)
    worker: str = None
    slot: str = None
    delay_risk: float = None
    delay_evidence: dict = field(default_factory=dict)


def load_csv(name):
    path = DATA_DIR / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_test_orders():
    return load_csv("test_orders.csv")


def number(row, key):
    value = (row.get(key) or "").strip()
    return float(value) if value else None


def apply_rules(plan, items):
    """Step 1: fuzzy urgency, then the expert system once per item line."""
    if plan.deadline_hours is not None and plan.order_value is not None:
        plan.urgency, _ = fuzzy_urgency(plan.deadline_hours, plan.order_value)

    engine = RuleEngine(RULES)

    for item_id in plan.item_ids:
        item = items.get(item_id)

        if item is None:
            plan.stock_holds.append(item_id)
            continue

        facts = {
            "order_id": plan.order_id,
            "urgent": plan.urgent,
            "stock": item.quantity,
            "required_quantity": 1,
            "reorder_level": REORDER_LEVEL,
        }

        final_facts, fired = engine.run(facts)

        if final_facts.get("priority") == "HIGH":
            plan.priority = "HIGH"
        if final_facts.get("stock_hold"):
            plan.stock_holds.append(item_id)
        if final_facts.get("reorder_alert"):
            plan.reorder_alerts.append(item_id)
            plan.reorder_cf[item_id], _, _ = reorder_advice(item.quantity, REORDER_LEVEL)

        for rule_name in fired:
            if rule_name not in plan.fired_rules:
                plan.fired_rules.append(rule_name)

    if plan.urgency is not None and plan.urgency >= HIGH_PRIORITY_SCORE:
        plan.priority = "HIGH"


def check_scans(plan, scans):
    """Step 2: is each item really on its shelf? (Bayes, Dempster-Shafer)"""
    for item_id in plan.item_ids:
        scan = scans.get(item_id)
        if scan is None or scan["last_scan"].strip().lower() != "failed":
            continue
        readings = [number(scan, k) for k in
                    ("rfid_on_shelf", "rfid_not_on_shelf",
                     "barcode_on_shelf", "barcode_not_on_shelf")]
        if all(r is not None for r in readings):
            rfid = {"on_shelf": readings[0], "not_on_shelf": readings[1],
                    "unknown": round(1 - readings[0] - readings[1], 10)}
            barcode = {"on_shelf": readings[2], "not_on_shelf": readings[3],
                       "unknown": round(1 - readings[2] - readings[3], 10)}
            belief = combine_evidence(rfid, barcode)["on_shelf"]
            action = "pick as planned" if belief >= 0.8 else "verify manually"
            plan.scan_checks[item_id] = (f"failed scan, but RFID + barcode give "
                                         f"{belief:.3f} belief on shelf: {action}")
        else:
            check = assess_scan(item_id)
            plan.scan_checks[item_id] = (f"failed scan: P(on shelf) = "
                                         f"{check['probability_on_shelf']:.3f}, "
                                         f"{check['recommendation']}")


def plan_route(plan, items, router):
    """Steps 3-4: item locations, A* route costs, picking order."""
    locations = [
        items[item_id].location_id
        for item_id in plan.item_ids
        if item_id in items and item_id not in plan.stock_holds
    ]

    if not locations:
        return

    plan.picking = plan_picking(locations, router.distance)
    plan.legs = [
        (a, b, router.route(a, b))
        for a, b in zip(plan.picking.order, plan.picking.order[1:])
    ]


def assign_workers(plans, workers):
    """Step 5: one CSP over every order that can be picked.

    MRV picks the most constrained order first; ties keep HIGH-priority
    orders first, and each order's options are sorted by time slot, so
    urgent orders get the earliest slots. Forward checking removes a used
    (worker, slot) from every other order at once.
    """
    pickable = sorted(
        (p for p in plans.values() if p.picking is not None),
        key=lambda p: (p.priority != "HIGH", p.order_id),
    )

    csp_orders = [scheduling_model.Order(p.order_id, [p.zone]) for p in pickable]
    csp_workers = [
        scheduling_model.Worker(w.worker_id, [w.zone], list(SHIFT_SLOTS))
        for w in workers.values()
        if w.available
    ]

    domains = scheduling_model.build_domains(csp_orders, csp_workers)
    result = find_schedule_mrv(csp_orders, domains, csp_workers)

    for order_id, (worker_id, slot) in result["schedule"].items():
        plans[order_id].worker = worker_id
        plans[order_id].slot = slot

    return result


def estimate_delay_risk(plans, workers):
    """Step 6: P(late dispatch) from the Bayesian network."""
    per_slot = {}
    for p in plans.values():
        if p.slot:
            per_slot[p.slot] = per_slot.get(p.slot, 0) + 1
    per_zone = {}
    for w in workers.values():
        if w.available:
            per_zone[w.zone] = per_zone.get(w.zone, 0) + 1

    for p in plans.values():
        if p.slot is None:
            continue
        p.delay_evidence = {
            "congestion": per_slot[p.slot] >= BUSY_SLOT_ORDERS,
            "shortage": per_zone.get(p.zone, 0) <= 1,
            "stockout": bool(p.stock_holds),
        }
        p.delay_risk = delay_risk(**p.delay_evidence)


def run_pipeline():
    """Plan every test order. Returns (plans, CSP result, workers, items, router)."""
    items = load_items(DATA_DIR / "inventory.csv")
    workers = load_workers(DATA_DIR / "workers.csv")
    router = GraphRouter.from_file(DATA_DIR / "warehouse.txt")
    scans = {row["item_id"].strip(): row for row in load_csv("scan_events.csv")}

    plans = {}
    for row in load_test_orders():
        plan = OrderPlan(
            order_id=row["order_id"].strip(),
            zone=row["zone"].strip(),
            urgent=row["priority"].strip().lower() == "urgent",
            item_ids=[i.strip() for i in row["item_ids"].split("|") if i.strip()],
            deadline_hours=number(row, "deadline_hours"),
            order_value=number(row, "order_value"),
        )
        apply_rules(plan, items)
        check_scans(plan, scans)
        plan_route(plan, items, router)
        plans[plan.order_id] = plan

    result = assign_workers(plans, workers)
    estimate_delay_risk(plans, workers)
    return plans, result, workers, items, router


def print_order_plan(plan, workers, items):
    print()
    print("=" * 60)
    print(f"ORDER {plan.order_id}   zone {plan.zone}   "
          f"{'urgent' if plan.urgent else 'normal'}")
    print("=" * 60)

    print()
    print("1. Rules (fuzzy urgency, expert system, certainty factors)")
    if plan.urgency is not None:
        print(f"   Fuzzy urgency  : {plan.urgency:.1f} / 100 "
              f"(due in {plan.deadline_hours:g} h, value Rs {plan.order_value:g}k)")
    print(f"   Priority       : {plan.priority}")
    print(f"   Rules fired    : {', '.join(plan.fired_rules) or 'none'}")
    if plan.stock_holds:
        print(f"   Stock hold     : {', '.join(plan.stock_holds)}")
    for item_id in plan.reorder_alerts:
        cf = plan.reorder_cf[item_id]
        print(f"   Reorder alert  : {item_id} {items[item_id].name}, "
              f"{items[item_id].quantity} left (level {REORDER_LEVEL}), "
              f"CF {cf:+.2f}: {reorder_label(cf)}")

    print()
    print("2. Item locations")
    for item_id in plan.item_ids:
        item = items.get(item_id)
        where = item.location_id if item else "not in inventory"
        name = item.name if item else ""
        print(f"   {item_id} {name:<18} -> {where}")
        if item_id in plan.scan_checks:
            print(f"        {plan.scan_checks[item_id]}")

    print()
    if plan.picking is None:
        print("3. Picking order: none, every item is on hold")
        print()
        print("4. Worker and time slot: not needed")
        return

    print(f"3. Picking order (TSP, {plan.picking.method})")
    print(f"   {' -> '.join(plan.picking.order)}")
    print(f"   Total distance : {plan.picking.cost} m")
    print("   Each leg (A* on the warehouse graph):")
    for a, b, leg in plan.legs:
        print(f"     {a:>3} -> {b:<4} {leg['cost']:>4} m   "
              f"{' -> '.join(leg['path'])}")

    print()
    print("4. Worker and time slot (CSP, MRV + forward checking)")
    if plan.worker is None:
        print("   No valid assignment with the current workers and slots")
        return
    worker = workers[plan.worker]
    print(f"   {plan.worker} {worker.name}, slot {plan.slot}")

    print()
    print("5. Delay risk (Bayesian network)")
    shown = ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in plan.delay_evidence.items())
    print(f"   Evidence       : {shown}")
    print(f"   P(late)        : {plan.delay_risk:.3f}  {risk_label(plan.delay_risk)}")


def print_summary(plans, result, workers):
    width = 108
    print()
    print("=" * width)
    print("PLAN FOR ALL TEST ORDERS")
    print("=" * width)
    print(f"{'Order':<8}{'Zone':<8}{'Urgency':>8}  {'Priority':<10}{'Picking order':<26}"
          f"{'Distance':>9}  {'Worker':<14}{'Slot':<7}{'Delay risk':<12}")
    print("-" * width)

    for plan in sorted(plans.values(), key=lambda p: (p.slot or "99", p.order_id)):
        if plan.picking is None:
            route, distance = "ON HOLD", ""
        else:
            route, distance = " -> ".join(plan.picking.order), f"{plan.picking.cost} m"
        worker = (f"{plan.worker} {workers[plan.worker].name}"
                  if plan.worker else "-")
        urgency = f"{plan.urgency:.0f}" if plan.urgency is not None else "-"
        risk = (f"{plan.delay_risk:.2f} {risk_label(plan.delay_risk)}"
                if plan.delay_risk is not None else "-")
        print(f"{plan.order_id:<8}{plan.zone:<8}{urgency:>8}  {plan.priority:<10}{route:<26}"
              f"{distance:>9}  {worker:<14}{plan.slot or '-':<7}{risk:<12}")

    print("-" * width)
    alerts = sorted({a for p in plans.values() for a in p.reorder_alerts})
    if alerts:
        print(f"Reorder alerts: {', '.join(alerts)}")
    checks = sorted({i for p in plans.values() for i in p.scan_checks})
    if checks:
        print(f"Failed scans checked: {', '.join(checks)}")
    if not result["success"]:
        print("CSP: no complete schedule.", end=" ")
        if result["unassigned"]:
            print("Orders with no possible worker:", ", ".join(result["unassigned"]))
        else:
            print("Every order has options, but their constraints conflict.")


# ============================================================
# Algorithm Demos
# ============================================================

def in_folder(folder, function):
    """Run a demo that opens files by relative path (../data/...) from its own folder."""
    import os
    here = os.getcwd()
    os.chdir(folder)
    try:
        function()
    finally:
        os.chdir(here)


def run_demo(name):
    from picking import ao_star, pallet_puzzle, aisle_game
    from rules import fol_rules, explain, certainty, fuzzy_priority
    import compare_solvers                        # scheduling/compare_solvers.py
    import hazmat_queens, lift_transfer, delay_network

    demos = {
        "routing": ("ROUTE SEARCH: BFS vs A* (grid)", None),
        "picking": ("PICKING ORDER: exact TSP vs nearest neighbour (grid A* costs)", None),
        "ao": ("AO* SEARCH: cheapest fulfilment plan", ao_star.run_demo),
        "csp": ("CSP: plain backtracking vs MRV + forward checking", lambda: in_folder(SCHEDULING_DIR, compare_solvers.run_comparison)),
        "pallet": ("TILES PROBLEM: pallet rearrangement with A*", pallet_puzzle.run_demo),
        "game": ("GAME PLAYING: minimax vs alpha-beta", aisle_game.run_demo),
        "queens": ("QUEENS PROBLEM: hazardous-pallet placement", lambda: hazmat_queens.solve_hazmat_queens(8)),
        "lift": ("MISSIONARIES AND CANNIBALS: lift transfer", lift_transfer.solve_lift_transfer),
        "fol": ("FIRST ORDER LOGIC: rules with variables", fol_rules.run_demo),
        "explain": ("EXPERT SYSTEM: explanation facility (WHY and HOW)", explain.run_demo),
        "certainty": ("CERTAINTY FACTORS: reorder and supplier-delay warnings", certainty.run_demo),
        "fuzzy": ("FUZZY LOGIC: graded order urgency", fuzzy_priority.run_demo),
        "bayesnet": ("BAYESIAN NETWORK: delay risk", delay_network.run_demo),
    }
    if name not in demos:
        raise SystemExit(f"Unknown demo '{name}'. Choose from: {', '.join(demos)}")

    if name == "routing":
        run_route_comparison("P1", "S9")       # prints its own heading
        return
    if name == "picking":
        run_tsp_demo()                         # prints its own heading
        return

    title, function = demos[name]
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    print()
    function()


DEMO_ORDER = ["routing", "picking", "ao", "csp", "pallet", "game", "queens", "lift",
              "fol", "explain", "certainty", "fuzzy", "bayesnet"]


# ============================================================
# Main Program
# ============================================================

def main(args):
    print()
    print("=" * 60)
    print("SMART DISTRIBUTION CENTER MANAGEMENT SYSTEM")
    print("=" * 60)

    if args and args[0] == "demos":
        for name in DEMO_ORDER:
            run_demo(name)
        return

    if args and args[0] == "demo":
        if len(args) < 2:
            raise SystemExit(f"Usage: python main.py demo <{'|'.join(DEMO_ORDER)}>")
        run_demo(args[1])
        return

    plans, result, workers, items, _ = run_pipeline()

    if args:
        order_id = args[0].upper()
        if order_id not in plans:
            raise SystemExit(f"Unknown order '{args[0]}'. "
                             f"Orders: {', '.join(plans)}")
        print_order_plan(plans[order_id], workers, items)
        print()
        return

    print_summary(plans, result, workers)
    print()
    print("Full plan for one order: python main.py ORD101")
    print("Algorithm demos:         python main.py demos")
    print()


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":
    main(sys.argv[1:])
