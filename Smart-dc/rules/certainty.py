"""
Certainty factors for rule-based reorder and supplier-delay warnings.

A plain rule says "reorder" or nothing. With certainty factors (CF, from
-1 = certainly false to +1 = certainly true, as in MYCIN) each piece of
evidence and each rule carries a confidence, and conclusions reached by
several rules are combined.

    CF(conclusion from one rule) = CF(rule) x CF(premise)
        where CF(A and B) = min(CF(A), CF(B)); a premise below 0.2 is ignored
    Two rules for the same conclusion, x and y:
        both > 0:  x + y(1 - x)
        both < 0:  x + y(1 + x)
        otherwise: (x + y) / (1 - min(|x|, |y|))

Rules (the rule CFs are the planning team's judgement, to be tuned):
    R1  stock low                          -> reorder needed     0.8
    R2  demand rising                      -> reorder needed     0.6
    R3  supplier late recently             -> supplier delay     0.7
    R4  supplier delay and stock low       -> reorder needed     0.9
    R5  stock high                         -> reorder needed    -0.7

Syllabus topic: Module 4 - Certainty factors and rule-based systems.

Run on its own (from the Smart-dc folder):   python -m rules.certainty
"""

THRESHOLD = 0.2

RULES = [
    ("R1", ["stock_low"], "reorder_needed", 0.8),
    ("R2", ["demand_rising"], "reorder_needed", 0.6),
    ("R3", ["supplier_late"], "supplier_delay", 0.7),
    ("R4", ["supplier_delay", "stock_low"], "reorder_needed", 0.9),
    ("R5", ["stock_high"], "reorder_needed", -0.7),
]


def combine(x, y):
    """Parallel combination of two CFs for the same conclusion."""
    if x >= 0 and y >= 0:
        return x + y * (1 - x)
    if x < 0 and y < 0:
        return x + y * (1 + x)
    return (x + y) / (1 - min(abs(x), abs(y)))


def stock_evidence(stock, reorder_level):
    """CFs for 'stock low' and 'stock high' from the stock level.

    Empty shelf: low = 1.0. At the reorder level: low = 0.5.
    At twice the level or more: low = 0 and high grows towards 1.
    """
    ratio = stock / reorder_level if reorder_level else 1.0
    low = max(-1.0, min(1.0, 1.0 - ratio / 2))
    high = max(0.0, min(1.0, (ratio - 2) / 3))
    return {"stock_low": round(max(low, 0.0), 3), "stock_high": round(high, 3)}


def infer(evidence, rules=RULES):
    """Forward chaining with CFs. Returns (cf per conclusion, trace lines)."""
    cf = dict(evidence)
    trace = []
    pending = list(rules)
    progress = True
    while progress:                     # a rule fires once all its premises are known
        progress = False
        for rule in list(pending):
            name, premises, conclusion, rule_cf = rule
            if not all(p in cf for p in premises):
                continue
            pending.remove(rule)
            progress = True
            premise_cf = min(cf[p] for p in premises)
            if premise_cf < THRESHOLD:
                trace.append(f"{name}: premise CF {premise_cf:.2f} below {THRESHOLD}, not used")
                continue
            new = rule_cf * premise_cf
            old = cf.get(conclusion)
            cf[conclusion] = new if old is None else combine(old, new)
            trace.append(f"{name}: {' and '.join(premises)} ({premise_cf:.2f}) x {rule_cf} = {new:+.2f}"
                         f" -> {conclusion} now {cf[conclusion]:+.2f}")
    return cf, trace


def reorder_advice(stock, reorder_level, demand_rising=0.0, supplier_late=0.0):
    evidence = stock_evidence(stock, reorder_level)
    evidence["demand_rising"] = demand_rising
    evidence["supplier_late"] = supplier_late
    cf, trace = infer(evidence)
    return cf.get("reorder_needed", 0.0), cf.get("supplier_delay", 0.0), trace


def label(cf):
    return ("reorder now" if cf >= 0.8 else "plan a reorder" if cf >= 0.5
            else "watch" if cf > 0 else "no reorder")


def run_demo():
    cases = [
        ("I010 Router: 8 left, supplier late", 8, 10, 0.3, 0.8),
        ("I004 Printer: 10 left, demand rising", 10, 10, 0.7, 0.0),
        ("I001 Keyboard: 20 left, nothing unusual", 20, 10, 0.0, 0.0),
        ("I014 Packing tape: 60 left, demand rising", 60, 10, 0.6, 0.0),
    ]
    print(f"{'Item':<44}{'CF reorder':>11}{'CF delay':>10}  Advice")
    for name, stock, level, demand, late in cases:
        reorder, delay, _ = reorder_advice(stock, level, demand, late)
        print(f"{name:<44}{reorder:>+11.2f}{delay:>+10.2f}  {label(reorder)}")
    print()
    print("Trace for I010:")
    for line in reorder_advice(8, 10, 0.3, 0.8)[2]:
        print("  " + line)


if __name__ == "__main__":
    run_demo()
