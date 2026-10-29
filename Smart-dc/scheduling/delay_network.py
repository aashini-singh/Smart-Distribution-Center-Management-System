"""
Delay-risk Bayesian network.

Three causes can make an order miss its dispatch time:

    Congestion ─┐
    Shortage  ──┼──> Late dispatch
    Stock-out ──┘

Congestion: the order's time slot is busy. Shortage: the order's zone
has only one available worker. Stock-out: a line of the order is on
stock hold. The three causes are independent; Late depends on all three
through a conditional probability table (CPT).

Inference is by enumeration: sum the joint probability over every
combination of the hidden variables, then normalise by the evidence.
The network answers both directions:
  - predictive: P(late | what we know about this order)
  - diagnostic: P(congestion | the order was late)

The probabilities below are assumed starting values for the simulated
warehouse. With real dispatch records they would be estimated from
counts.

Syllabus topic: Module 4 - Bayesian networks.

Run on its own (from the scheduling folder):   python delay_network.py
"""

from itertools import product

CAUSES = ["congestion", "shortage", "stockout"]

PRIOR = {"congestion": 0.30, "shortage": 0.20, "stockout": 0.10}

# P(late = True | congestion, shortage, stockout)
CPT_LATE = {
    (False, False, False): 0.05,
    (True,  False, False): 0.30,
    (False, True,  False): 0.25,
    (False, False, True):  0.60,
    (True,  True,  False): 0.55,
    (True,  False, True):  0.75,
    (False, True,  True):  0.70,
    (True,  True,  True):  0.90,
}


def joint(values):
    """P(congestion, shortage, stockout, late) for one full assignment."""
    p = 1.0
    for cause in CAUSES:
        p *= PRIOR[cause] if values[cause] else 1 - PRIOR[cause]
    p_late = CPT_LATE[tuple(values[c] for c in CAUSES)]
    return p * (p_late if values["late"] else 1 - p_late)


def probability(query, evidence=None):
    """P(query | evidence) by enumeration. query and evidence: {variable: bool}."""
    evidence = evidence or {}
    variables = CAUSES + ["late"]
    numerator = denominator = 0.0
    for combo in product([True, False], repeat=len(variables)):
        values = dict(zip(variables, combo))
        if any(values[k] != v for k, v in evidence.items()):
            continue
        p = joint(values)
        denominator += p
        if all(values[k] == v for k, v in query.items()):
            numerator += p
    return numerator / denominator


def delay_risk(congestion=None, shortage=None, stockout=None):
    """P(late) given whatever is known; None means unknown."""
    evidence = {k: v for k, v in
                {"congestion": congestion, "shortage": shortage, "stockout": stockout}.items()
                if v is not None}
    return probability({"late": True}, evidence)


def risk_label(p):
    return "HIGH" if p >= 0.5 else "MEDIUM" if p >= 0.2 else "LOW"


def run_demo():
    print(f"P(late) with no evidence: {delay_risk():.3f}")
    print()
    print("Predictive (cause known -> risk of a late dispatch):")
    cases = [
        ("Quiet slot, staffed zone, all in stock", dict(congestion=False, shortage=False, stockout=False)),
        ("Busy slot only", dict(congestion=True)),
        ("Busy slot and one worker in zone", dict(congestion=True, shortage=True)),
        ("Stock-out on one line", dict(stockout=True)),
        ("All three causes", dict(congestion=True, shortage=True, stockout=True)),
    ]
    for name, ev in cases:
        p = delay_risk(**ev)
        print(f"  {name:<42}{p:>6.3f}  {risk_label(p)}")
    print()
    print("Diagnostic (the order was late -> which cause?):")
    for cause in CAUSES:
        before = PRIOR[cause]
        after = probability({cause: True}, {"late": True})
        print(f"  P({cause} | late) = {after:.3f}   (prior {before:.2f})")


if __name__ == "__main__":
    run_demo()
