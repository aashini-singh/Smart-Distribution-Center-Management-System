"""
Explanation facility of the expert system.

An expert system has four parts:

    knowledge base     rules.py (IF-THEN rules) and fol_rules.py
    working memory     the facts about one order (a dict, or a fact set)
    inference engine   engine.py (forward chaining) and fol_rules.forward_chain
    explanation        this file: WHY a rule fired, HOW a fact was derived

why(facts)    runs the propositional engine and gives, for every rule that
              fired, the facts it read and the conclusion it added.
how(fact, kb) walks back through the first-order derivations to the
              original facts, as an indented proof tree.

Syllabus topic: Module 3 - Expert system, architecture of expert system.

Run on its own (from the Smart-dc folder):   python -m rules.explain
"""

from rules.engine import RuleEngine
from rules.rules import RULES
from rules import fol_rules

# For each propositional rule: the facts it reads and how to say why it fired.
REASONS = {
    "High Priority Order": (["urgent"], lambda f: "the order is marked urgent"),
    "Stock Hold": (["stock", "required_quantity"],
                   lambda f: f"stock {f['stock']} is below the {f['required_quantity']} required"),
    "Reorder Alert": (["stock", "reorder_level"],
                      lambda f: f"stock {f['stock']} is at or below the reorder level {f['reorder_level']}"),
}


def why(facts, rules=RULES):
    """Run the engine and return (final_facts, list of explanation lines)."""
    final, fired = RuleEngine(rules).run(facts)
    lines = []
    for name in fired:
        rule = next(r for r in rules if r.name == name)
        added = rule.action(final)
        reads, reason = REASONS.get(name, ([], lambda f: "its condition held"))
        concluded = ", ".join(f"{k} = {v}" for k, v in added.items())
        used = ", ".join(f"{k} = {facts.get(k)}" for k in reads)
        lines.append(f"{name}: fired because {reason(final)} [{used}], so {concluded}")
    if not fired:
        lines.append("No rule fired: the order needs no special handling.")
    return final, lines


def how(fact, kb, depth=0):
    """Proof tree for a first-order fact: the rule and the premises below it."""
    pad = "    " * depth
    show = f"{fact[0]}({', '.join(map(str, fact[1:]))})"
    d = kb.derivations.get(fact)
    if d is None:
        return [f"{pad}{show}   [given fact]"]
    bound = ", ".join(f"{k[1:]}={v}" for k, v in d.bindings.items())
    lines = [f"{pad}{show}   [by {d.rule}, with {bound}]"]
    for premise in d.premises:
        lines.extend(how(premise, kb, depth + 1))
    return lines


def run_demo():
    print("WHY: propositional rules for one order line")
    facts = {"order_id": "ORD105", "urgent": False, "stock": 8,
             "required_quantity": 1, "reorder_level": 10}
    print(f"  Working memory: {facts}")
    _, lines = why(facts)
    for line in lines:
        print("  -", line)
    print()

    kb = fol_rules.load_kb()
    fol_rules.forward_chain(kb, fol_rules.RULES)
    for fact in [("ReorderAlert", "ORD105", "I010"), ("NeedsProtection", "ORD106")]:
        print(f"HOW was {fact[0]}({', '.join(fact[1:])}) derived?")
        for line in how(fact, kb):
            print("  " + line)
        print()


if __name__ == "__main__":
    run_demo()
