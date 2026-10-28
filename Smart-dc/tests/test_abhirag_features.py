"""
Tests for Abhirag's features: first-order rules, explanation facility,
certainty factors and fuzzy urgency.

Run from the Smart-dc folder:
    python -m unittest tests.test_abhirag_features -v
"""

import csv
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rules import fol_rules as fol
from rules import explain
from rules import certainty as cf
from rules import fuzzy_priority as fz


def rows(name):
    with (ROOT / "data" / name).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


class FirstOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = fol.load_kb()
        fol.forward_chain(cls.kb, fol.RULES)

    def derived(self, predicate):
        return {f[1:] for f in self.kb.facts if f[0] == predicate}

    def test_unify(self):
        self.assertEqual(fol.unify(("P", "?x", "a"), ("P", 1, "a"), {}), {"?x": 1})
        self.assertIsNone(fol.unify(("P", "?x", "?x"), ("P", 1, 2), {}))
        self.assertIsNone(fol.unify(("P", "a"), ("Q", "a"), {}))

    def test_priority_matches_urgent_orders(self):
        urgent = {r["order_id"] for r in rows("test_orders.csv") if r["priority"] == "urgent"}
        self.assertEqual({o for o, _ in self.derived("Priority")}, urgent)

    def test_reorder_alerts_match_a_direct_count(self):
        stock = {r["item_id"]: int(r["quantity"]) for r in rows("inventory.csv")}
        expected = {(r["order_id"], i) for r in rows("test_orders.csv")
                    for i in r["item_ids"].split("|") if stock[i] <= fol.REORDER_LEVEL}
        self.assertEqual(self.derived("ReorderAlert"), expected)

    def test_fragile_orders_use_the_semantic_net(self):
        fragile = {r["subject"] for r in rows("relations.csv")
                   if r["relation"] == "is-a" and r["object"] == "fragile"}
        expected = {(r["order_id"],) for r in rows("test_orders.csv")
                    if fragile & set(r["item_ids"].split("|"))}
        self.assertEqual(self.derived("NeedsProtection"), expected)

    def test_rules_chain_on_derived_facts(self):
        kb = fol.KnowledgeBase()
        kb.add("Urgent", "X1"); kb.add("Contains", "X1", "G1"); kb.add("IsA", "G1", "fragile")
        fol.forward_chain(kb, fol.RULES)
        self.assertIn(("PackFirst", "X1"), kb.facts)

    def test_every_derived_fact_has_a_proof_to_given_facts(self):
        for fact in self.kb.derivations:
            leaves = [line for line in explain.how(fact, self.kb) if "[given fact]" in line]
            self.assertTrue(leaves)


class ExplanationTests(unittest.TestCase):
    def test_why_lists_each_fired_rule_with_its_reason(self):
        facts = {"urgent": True, "stock": 3, "required_quantity": 5, "reorder_level": 4}
        final, lines = explain.why(facts)
        self.assertEqual(len(lines), 3)
        self.assertTrue(any("urgent" in l for l in lines))
        self.assertTrue(any("below the 5 required" in l for l in lines))
        self.assertEqual(final["priority"], "HIGH")

    def test_why_says_when_nothing_fired(self):
        _, lines = explain.why({"urgent": False, "stock": 50,
                                "required_quantity": 1, "reorder_level": 10})
        self.assertIn("No rule fired", lines[0])


class CertaintyTests(unittest.TestCase):
    def test_combination_rules(self):
        self.assertAlmostEqual(cf.combine(0.6, 0.5), 0.8)
        self.assertAlmostEqual(cf.combine(-0.6, -0.5), -0.8)
        self.assertAlmostEqual(cf.combine(0.6, -0.4), 0.2 / 0.6)

    def test_combination_is_symmetric_and_bounded(self):
        rng = random.Random(4)
        for _ in range(500):
            x, y = rng.uniform(-0.99, 0.99), rng.uniform(-0.99, 0.99)
            self.assertAlmostEqual(cf.combine(x, y), cf.combine(y, x))
            self.assertLessEqual(abs(cf.combine(x, y)), 1.0)

    def test_more_evidence_raises_reorder_confidence(self):
        alone, _, _ = cf.reorder_advice(8, 10)
        with_late, _, _ = cf.reorder_advice(8, 10, supplier_late=0.8)
        self.assertGreater(with_late, alone)

    def test_full_stock_argues_against_reorder(self):
        self.assertLess(cf.reorder_advice(60, 10)[0], 0)
        self.assertEqual(cf.reorder_advice(20, 10)[0], 0)


class FuzzyTests(unittest.TestCase):
    # Centroid defuzzification is not strictly monotone: when one rule's
    # clipped set shrinks, the centroid can wobble. The largest dip over
    # the whole input grid is under 1 point, so the trend tests allow 1.
    WOBBLE = 1.0

    def test_score_stays_in_range(self):
        for h in range(0, 73, 3):
            for v in range(0, 101, 10):
                self.assertTrue(0 <= fz.urgency(h, v)[0] <= 100)

    def test_closer_deadline_never_lowers_urgency(self):
        for v in (10, 40, 80):
            scores = [fz.urgency(h, v)[0] for h in range(0, 73)]
            self.assertGreater(scores[0], scores[-1] + 40)      # clear overall fall
            for later, sooner in zip(scores[1:], scores):
                self.assertLessEqual(later, sooner + self.WOBBLE)

    def test_higher_value_never_lowers_urgency(self):
        for h in (3, 20, 48):
            scores = [fz.urgency(h, v)[0] for v in range(0, 101)]
            for a, b in zip(scores, scores[1:]):
                self.assertLessEqual(a, b + self.WOBBLE)

    def test_memberships(self):
        self.assertEqual(fz.trapezoid(5, 0, 0, 4, 12), 0.875)
        self.assertEqual(fz.trapezoid(0, 0, 0, 4, 12), 1.0)
        self.assertEqual(fz.trapezoid(72, 24, 40, 72, 72), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
