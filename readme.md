# Smart Distribution Center Management System

AI planner for one warehouse. For every order it decides the priority, finds the items, plans the shortest picking tour, assigns a worker and a time slot, and estimates the risk of a late dispatch. Each decision uses a topic from the AI syllabus (CCSAI0301, Modules 1–4).

Pure Python 3 (standard library only). No packages to install.

---

## Quick start

```bash
cd Smart-dc
python main.py              # manager console (interactive menu)
python main.py plan         # plan for all pending orders
python main.py ORD104       # every step for one order
python main.py demos        # all algorithm demos and comparisons
python main.py demo game    # one demo
python -m unittest discover -s tests   # 52 tests
```

Demo names: `routing, picking, ao, csp, pallet, game, queens, lift, fol, explain, certainty, fuzzy, bayesnet`.

---

## How one order is planned

```
Order → Rules → Item lookup → Route costs → Picking tour → Worker & slot → Final plan
```

| Step | What happens | Technique | Code |
|---|---|---|---|
| 1. Rules | Urgency score from due time and value; rules set the priority | Fuzzy logic, expert system (forward chaining), certainty factors | `rules/` |
| 2. Item lookup | Which shelf holds each item; is it really there after a failed scan | Frames, semantic net, Bayes, Dempster–Shafer | `warehouse/`, `uncertainty/` |
| 3. Route costs | Shortest route between every pair of stops | A* (admissible heuristic: Manhattan × 2/3) | `routing/`, `picking/route_costs.py` |
| 4. Picking tour | Best order to visit the shelves, P1 → items → D1 | TSP: exact up to 8 stops, else nearest neighbour | `picking/tsp_*.py` |
| 5. Worker & slot | One worker of the right zone, one free slot, urgent orders first | CSP, backtracking with MRV + forward checking | `scheduling/` |
| 6. Delay risk | P(late dispatch) | Bayesian network (enumeration) | `scheduling/delay_network.py` |

Example (`python main.py ORD104`): urgency 92.2 → HIGH; I007 on S7 (failed scan, P(on shelf) = 0.628, verify manually), I008 on S8; tour P1 → S7 → S8 → D1 = 100 m; W004 Neha, 09-10; P(late) = 0.55.

Orders that cannot be planned (aisle blocked, no worker, no free slot) go to a waiting list with the reason instead of crashing.

---

## Current plan (`python main.py plan`)

| Order | Priority | Picking tour | Distance | Worker | Slot |
|---|---|---|---|---|---|
| ORD101 | HIGH | P1 S2 S1 D1 | 40 m | W001 Rahul | 09-10 |
| ORD102 | NORMAL | P1 S3 S4 D1 | 70 m | W002 Priya | 09-10 |
| ORD104 | HIGH | P1 S7 S8 D1 | 100 m | W004 Neha | 09-10 |
| ORD105 | NORMAL | P1 S9 S1 D1 | 120 m | W005 Karan | 09-10 |
| ORD107 | HIGH | P1 L03 L04 D1 | 150 m | W006 Meera | 09-10 |
| ORD103 | NORMAL | P1 S6 S5 D1 | 70 m | W006 Meera | 10-11 |
| ORD106 | NORMAL | P1 L01 L02 D1 | 110 m | W002 Priya | 10-11 |
| ORD108 | NORMAL | P1 L05 L06 D1 | 190 m | W004 Neha | 10-11 |
| ORD109 | NORMAL | P1 L07 L08 D1 | 230 m | W001 Rahul | 10-11 |
| ORD110 | NORMAL | P1 L09 L10 D1 | 150 m | W002 Priya | 11-12 |

Data: 32 locations, 20 items, 6 workers (W003 Arjun absent, so 5 available), 10 orders. Reorder alerts: I004, I010.

---

## Manager console (`python main.py`)

| # | Option | # | Option |
|---|---|---|---|
| 1 | Add a new order (planned at once) | 10 | Mark a worker absent / present |
| 2 | Plan all orders (MRV vs plain backtracking) | 11 | Robot aisle game (minimax / alpha-beta) |
| 3 | Show one order's plan | 12 | Rearrange a pallet block (8-puzzle, A*) |
| 4 | Find a route between two locations | 13 | Place hazardous pallets (N-Queens) |
| 5 | Block / reopen an aisle | 14 | Lift transfer (missionaries–cannibals) |
| 6 | Compare search algorithms | 15 | Measure liquid (water-jug, BFS) |
| 7 | Stock and reorder alerts | 16 | Cheapest sources for an order (AO*) |
| 8 | Failed scan check (Bayes / Dempster–Shafer) | 17 | Which algorithm is used for what |
| 9 | Knowledge base queries (semantic net, rules) | 0 | Exit |

The best method is used for each task; the weaker ones stay available for comparison.

---

## Verified results

| Topic | Result |
|---|---|
| Search, 10 location pairs (avg nodes expanded) | IDS 555.6, BFS 32.2, DFS 22.3, Bidirectional 15.8, A* 12.4, Greedy 7.5. DFS shortest on 3/10; A* matched Dijkstra on every pair |
| Alpha-beta vs minimax | Same value; 16 pallets: 7,534 vs 131,071 nodes (−94%). 8-pallet demo aisle: game value 17 |
| TSP demo order (S5, S5, S8) | Exact 100 m vs nearest neighbour 130 m |
| 8-puzzle (pallet block) | 17 moves; Manhattan 113 vs misplaced 883 nodes |
| AO* | 9 of 14 nodes expanded, cost 260 m |
| Water-jug / lift | 4 steps (BFS) / 11 journeys |
| Bayes, one failed scan | P(on shelf) 0.95 → 0.628 → manual check |
| Dempster–Shafer, RFID + barcode | 0.856 |
| Bayesian network | P(late) 0.218 normally, 0.900 with congestion and shortage |
| First-order logic | 160 facts → 23 derived |

---

## Folder structure

```
Smart-dc/
├── main.py            pipeline + command line
├── console.py         manager console
├── data/              warehouse graph, items, workers, orders, relations, scans
├── warehouse/         graph, frames (models.py), loaders, semantic net
├── routing/           BFS, DFS, IDS, bidirectional, greedy, A*, comparison, water-jug
├── picking/           route costs, TSP exact / nearest, AO*, 8-puzzle, aisle game
├── scheduling/        CSP (plain + MRV), delay Bayesian network, N-Queens, lift
├── rules/             rule engine, rules, FOL rules, explanation, certainty factors, fuzzy
├── uncertainty/       Bayes scan check, Dempster–Shafer
├── tests/             52 unit tests (each algorithm checked against a simpler method)
└── results/           saved outputs
```

---

## Syllabus coverage

| Module | Topics used |
|---|---|
| 1 | State space, BFS, DFS, IDS, bidirectional, heuristics, greedy best-first, A*, AO*, CSP |
| 2 | TSP, 8-puzzle, water-jug, N-Queens, missionaries–cannibals, minimax, alpha-beta |
| 3 | Frames, semantic net, propositional logic, first-order logic and unification, forward chaining, expert system with explanation |
| 4 | Bayes' theorem, Bayesian network, Dempster–Shafer, certainty factors, fuzzy logic |

---

## Future scope

- Live RFID / IoT scan readings
- Several robots routed without collisions
- A web dashboard instead of the console

---

## Project information

| Field | Details |
|---|---|
| Program | B.Tech CSE |
| Course | Artificial Intelligence (CCSAI0301) |
| Faculty | Dr. Mohd. Nazim |
| Assignment | Group PBL Project, Group 3 |
| SDG | SDG 9 – Industry, Innovation and Infrastructure |

## Team

- Aashini Singh
- Amulya Pratap Singh
- Aditya Upadhyay
- Ananya Rastogi
- Abhirag Verma

## References

1. Araújo, Vale, Longras, Dias & Rocha (2026), *Warehouse Layout Improvement and Routing Optimization*.
2. Ho, Tang, Lee, Tam & Chow (2026), *Toward Industry 5.0: An IoT-Enabled Digital Twin*.
3. Arvind, Shrinidhi, Deepa & Maheedhar (2025), *Intelligent Warehousing*.
4. Tsarouhas & Papaevangelou (2024), *Critical Steps and Conditions*.
5. de Koster, Le-Duc & Roodbergen (2007), *Design and control of warehouse order picking*, EJOR.
