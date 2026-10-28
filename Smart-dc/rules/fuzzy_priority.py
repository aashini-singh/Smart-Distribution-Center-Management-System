"""
Fuzzy order urgency.

"Urgent" or "normal" is a hard line: one hour either side of a cut-off
flips the label. Fuzzy logic gives a graded score from two inputs:

    hours to deadline   0-72 h        sets: near, medium, far
    order value         0-100 (Rs k)  sets: low, medium, high
    urgency (output)    0-100         sets: low, normal, high, critical

Mamdani inference: each rule fires to the degree of its weakest input
(AND = min), clips its output set, the clipped sets are joined (max), and
the centroid of the result is the urgency score.

Syllabus topic: Module 4 - Fuzzy logic.

Run on its own (from the Smart-dc folder):   python -m rules.fuzzy_priority
"""


def trapezoid(x, a, b, c, d):
    """0 below a, rising to 1 at b, flat to c, falling to 0 at d."""
    if x <= a or x >= d:
        return 1.0 if (x <= a and a == b) or (x >= d and c == d) else 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


DEADLINE = {"near": (0, 0, 4, 12), "medium": (6, 16, 24, 36), "far": (24, 40, 72, 72)}
VALUE = {"low": (0, 0, 10, 30), "medium": (15, 35, 50, 70), "high": (50, 75, 100, 100)}
URGENCY = {"low": (0, 0, 15, 35), "normal": (20, 40, 50, 65),
           "high": (55, 70, 78, 90), "critical": (80, 92, 100, 100)}

RULES = [
    ("near", "high", "critical"),
    ("near", "medium", "critical"),
    ("near", "low", "high"),
    ("medium", "high", "high"),
    ("medium", "medium", "normal"),
    ("medium", "low", "normal"),
    ("far", "high", "normal"),
    ("far", "medium", "low"),
    ("far", "low", "low"),
]

HIGH_PRIORITY_SCORE = 70


def fuzzify(hours, value):
    hours = min(max(hours, 0), 72)
    value = min(max(value, 0), 100)
    return ({k: trapezoid(hours, *p) for k, p in DEADLINE.items()},
            {k: trapezoid(value, *p) for k, p in VALUE.items()})


def urgency(hours, value, step=0.5):
    """Return (score 0-100, rule strengths)."""
    d, v = fuzzify(hours, value)
    strength = {}
    for dl, vl, out in RULES:
        s = min(d[dl], v[vl])
        strength[out] = max(strength.get(out, 0.0), s)
    num = den = 0.0
    x = 0.0
    while x <= 100:
        mu = max(min(s, trapezoid(x, *URGENCY[out])) for out, s in strength.items())
        num += x * mu
        den += mu
        x += step
    return (num / den if den else 0.0), strength


def label(score):
    return ("critical" if score >= 85 else "high" if score >= HIGH_PRIORITY_SCORE
            else "normal" if score >= 35 else "low")


def run_demo():
    print(f"{'Hours to deadline':>18}{'Value (Rs k)':>14}{'Urgency':>10}  Label")
    for hours, value in [(3, 80), (7, 40), (9, 40), (20, 90), (20, 30), (48, 90), (60, 10)]:
        score, _ = urgency(hours, value)
        print(f"{hours:>18}{value:>14}{score:>10.1f}  {label(score)}")
    print()
    print("Same order value (Rs 40k), deadline moving away two hours at a time:")
    sweep = [(h, urgency(h, 40)[0]) for h in range(4, 17, 2)]
    print("  " + "   ".join(f"{h} h: {s:.0f}" for h, s in sweep))
    print("The score falls in steps as the deadline moves away, instead of")
    print("switching from urgent to normal at a single cut-off.")


if __name__ == "__main__":
    run_demo()
