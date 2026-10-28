"""
First-order rules with variables, run by forward chaining.

The propositional rules in rules.py talk about one order at a time
("the order is urgent"). First-order rules use variables, so one rule
covers every order and every item:

    Urgent(?o)                                  -> Priority(?o, HIGH)
    Stock(?i, ?q) & ReorderLevel(?i, ?l) & ?q <= ?l -> LowStock(?i)
    Contains(?o, ?i) & LowStock(?i)             -> ReorderAlert(?o, ?i)
    Contains(?o, ?i) & IsA(?i, fragile)         -> NeedsProtection(?o)
    Contains(?o, ?i) & StoredIn(?i, ?s) & LocatedIn(?s, ?z)
        & OrderZone(?o, ?oz) & ?z != ?oz        -> CrossZone(?o, ?i)
    Priority(?o, HIGH) & NeedsProtection(?o)    -> PackFirst(?o)

Facts are tuples such as ("Contains", "ORD101", "I001"). A pattern holds
variables (names starting with ?). Matching a rule means finding every
substitution of its variables that makes all premises known facts
(unification against the fact base). New facts are added until no rule
produces anything new. Every derived fact remembers the rule, the
substitution and the premises that produced it, which the explanation
facility in explain.py uses.

Syllabus topic: Module 3 - First order logic.

Run on its own (from the Smart-dc folder):   python -m rules.fol_rules
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
REORDER_LEVEL = 10            # same value main.py uses


def is_var(term):
    return isinstance(term, str) and term.startswith("?")


def unify(pattern, fact, bindings):
    """Extend bindings so pattern equals fact, or return None."""
    if len(pattern) != len(fact):
        return None
    new = dict(bindings)
    for p, f in zip(pattern, fact):
        if is_var(p):
            if p in new and new[p] != f:
                return None
            new[p] = f
        elif p != f:
            return None
    return new


def substitute(pattern, bindings):
    return tuple(bindings.get(t, t) if is_var(t) else t for t in pattern)


@dataclass
class FOLRule:
    name: str
    premises: list                     # patterns, all must hold
    conclusion: tuple                  # pattern added for each match
    test: object = None                # optional check on the bindings
    text: str = ""                     # the rule in logic notation


@dataclass
class Derivation:
    rule: str
    bindings: dict
    premises: list


@dataclass
class KnowledgeBase:
    facts: set = field(default_factory=set)
    derivations: dict = field(default_factory=dict)   # fact -> Derivation

    def add(self, *fact):
        self.facts.add(tuple(fact))

    def query(self, pattern):
        """Every binding set that makes the pattern a known fact."""
        return [b for f in self.facts if (b := unify(pattern, f, {})) is not None]


def matches(premises, kb, bindings=None):
    """All substitutions satisfying every premise (a left-to-right join)."""
    bindings = bindings or {}
    if not premises:
        yield bindings
        return
    first, rest = premises[0], premises[1:]
    for fact in list(kb.facts):
        b = unify(first, fact, bindings)
        if b is not None:
            yield from matches(rest, kb, b)


def forward_chain(kb, rules):
    """Fire rules until no new fact appears. Returns the facts added."""
    added = []
    changed = True
    while changed:
        changed = False
        for rule in rules:
            for b in list(matches(rule.premises, kb)):
                if rule.test is not None and not rule.test(b):
                    continue
                new = substitute(rule.conclusion, b)
                if new not in kb.facts:
                    kb.facts.add(new)
                    kb.derivations[new] = Derivation(
                        rule.name, b, [substitute(p, b) for p in rule.premises])
                    added.append(new)
                    changed = True
    return added


RULES = [
    FOLRule("R1 Urgent orders", [("Urgent", "?o")], ("Priority", "?o", "HIGH"),
            text="∀o Urgent(o) → Priority(o, HIGH)"),
    FOLRule("R2 Low stock", [("Stock", "?i", "?q"), ("ReorderLevel", "?i", "?l")],
            ("LowStock", "?i"), test=lambda b: b["?q"] <= b["?l"],
            text="∀i,q,l Stock(i,q) ∧ ReorderLevel(i,l) ∧ q ≤ l → LowStock(i)"),
    FOLRule("R3 Reorder alert", [("Contains", "?o", "?i"), ("LowStock", "?i")],
            ("ReorderAlert", "?o", "?i"),
            text="∀o,i Contains(o,i) ∧ LowStock(i) → ReorderAlert(o,i)"),
    FOLRule("R4 Fragile goods", [("Contains", "?o", "?i"), ("IsA", "?i", "fragile")],
            ("NeedsProtection", "?o"),
            text="∀o,i Contains(o,i) ∧ IsA(i, fragile) → NeedsProtection(o)"),
    FOLRule("R5 Cross-zone pick",
            [("Contains", "?o", "?i"), ("StoredIn", "?i", "?s"),
             ("LocatedIn", "?s", "?z"), ("OrderZone", "?o", "?oz")],
            ("CrossZone", "?o", "?i"), test=lambda b: b["?z"] != b["?oz"],
            text="∀o,i,s,z,oz Contains(o,i) ∧ StoredIn(i,s) ∧ LocatedIn(s,z) ∧ OrderZone(o,oz) ∧ z ≠ oz → CrossZone(o,i)"),
    FOLRule("R6 Pack first", [("Priority", "?o", "HIGH"), ("NeedsProtection", "?o")],
            ("PackFirst", "?o"),
            text="∀o Priority(o, HIGH) ∧ NeedsProtection(o) → PackFirst(o)"),
]


def load_kb(data_dir=DATA):
    """Facts from inventory.csv, relations.csv and test_orders.csv."""
    kb = KnowledgeBase()
    with (data_dir / "inventory.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            kb.add("Stock", row["item_id"], int(row["quantity"]))
            kb.add("ReorderLevel", row["item_id"], REORDER_LEVEL)
    with (data_dir / "relations.csv").open(newline="", encoding="utf-8-sig") as f:
        names = {"is-a": "IsA", "stored-in": "StoredIn", "located-in": "LocatedIn"}
        for row in csv.DictReader(f):
            rel = names.get(row["relation"].strip().lower())
            if rel:
                kb.add(rel, row["subject"].strip(), row["object"].strip())
    with (data_dir / "test_orders.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            o = row["order_id"].strip()
            kb.add("Order", o)
            kb.add("OrderZone", o, row["zone"].strip())
            if row["priority"].strip().lower() == "urgent":
                kb.add("Urgent", o)
            for item in row["item_ids"].split("|"):
                if item.strip():
                    kb.add("Contains", o, item.strip())
    return kb


def run_demo():
    kb = load_kb()
    start = len(kb.facts)
    added = forward_chain(kb, RULES)
    print("Rules (one rule covers every order and item):")
    for r in RULES:
        print(f"  {r.name:<20}{r.text}")
    print()
    print(f"Facts loaded: {start}   facts derived: {len(added)}")
    for predicate in ["Priority", "ReorderAlert", "NeedsProtection", "PackFirst", "CrossZone"]:
        found = sorted(f[1:] for f in added if f[0] == predicate)
        shown = ", ".join("(" + ", ".join(map(str, x)) + ")" for x in found)
        print(f"  {predicate:<16}{len(found):>3}  {shown}")
    print()
    print("Query  ?- NeedsProtection(?o):",
          sorted(b["?o"] for b in kb.query(("NeedsProtection", "?o"))))


if __name__ == "__main__":
    run_demo()
