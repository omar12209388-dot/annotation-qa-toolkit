import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotation_qa.agreement import cohen_kappa, fleiss_kappa, krippendorff_alpha
from annotation_qa.audit import Annotation, audit_annotators
from annotation_qa.pii import find_pii, redact


class TestAgreement(unittest.TestCase):
    def test_cohen_perfect(self):
        self.assertEqual(cohen_kappa("aab", "aab"), 1.0)

    def test_cohen_chance(self):
        # 50% observed agreement at 50% expected chance -> kappa 0
        self.assertAlmostEqual(cohen_kappa("aabb", "abab"), 0.0)

    def test_fleiss_perfect(self):
        self.assertEqual(fleiss_kappa([[3, 0], [3, 0], [0, 3]]), 1.0)

    def test_alpha_with_missing(self):
        a = ["x", "x", "y", None]
        b = ["x", "y", "y", "y"]
        c = ["x", "x", "y", "y"]
        alpha = krippendorff_alpha(a, b, c)
        self.assertGreater(alpha, 0.5)
        self.assertLessEqual(alpha, 1.0)

    # Reference values cross-checked against the statsmodels and
    # `krippendorff` packages during development (diff < 1e-6).

    def test_fleiss_reference_values(self):
        m1 = [[8, 2], [4, 6], [10, 0], [3, 7], [5, 5], [9, 1]]
        self.assertAlmostEqual(fleiss_kappa(m1), 0.226699, places=5)
        m2 = [[5, 3, 2], [1, 8, 1], [0, 4, 6], [7, 2, 1], [2, 2, 6], [0, 0, 10]]
        self.assertAlmostEqual(fleiss_kappa(m2), 0.298546, places=5)

    def test_cohen_reference_values(self):
        a = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
        b = [1, 1, 1, 0, 0, 1, 0, 0, 1, 1]
        self.assertAlmostEqual(cohen_kappa(a, b), 0.4, places=6)
        c = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
        d = [1, 2, 3, 1, 2, 2, 1, 2, 3, 3]
        self.assertAlmostEqual(cohen_kappa(c, d), 0.701493, places=5)

    def test_alpha_reference_values(self):
        r1 = [[1, 1, 2, 2, 3, 3, 1, 2],
              [1, 2, 2, 2, 3, 3, 1, 2],
              [1, 1, 2, None, 3, 3, 2, 2]]
        self.assertAlmostEqual(krippendorff_alpha(*r1), 0.744186, places=5)
        r2 = [[1, 2, 1, 2], [1, 2, 2, 1], [2, 2, 1, 2], [1, 1, 2, 2]]
        self.assertAlmostEqual(krippendorff_alpha(*r2), -0.031746, places=5)


class TestAudit(unittest.TestCase):
    def test_flags_low_accuracy(self):
        anns = [
            Annotation(f"i{k}", "good", "safe") for k in range(10)
        ] + [
            Annotation(f"i{k}", "bad", "safe" if k % 2 else "violating")
            for k in range(10)
        ]
        gold = {f"i{k}": "safe" for k in range(10)}
        audits = {a.annotator: a for a in audit_annotators(anns, gold)}
        self.assertFalse(audits["good"].flagged)
        self.assertTrue(audits["bad"].flagged)
        self.assertEqual(audits["bad"].accuracy, 0.5)


class TestPii(unittest.TestCase):
    def test_email_and_card(self):
        hits = find_pii("reach me at mona.example@mail.com or 4111 1111 1111 1111")
        kinds = {h.kind for h in hits}
        self.assertIn("email", kinds)
        self.assertIn("credit_card", kinds)

    def test_bad_card_rejected_by_luhn(self):
        hits = find_pii("card 4111 1111 1111 1112 here")
        self.assertNotIn("credit_card", {h.kind for h in hits})

    def test_eastern_arabic_digits_id(self):
        hits = find_pii("الرقم القومي ٢٩٠١٢٠١١٢٣٤٥٦٧")
        self.assertIn("egyptian_national_id", {h.kind for h in hits})

    def test_redact(self):
        out = redact("mail mona.example@mail.com please")
        self.assertIn("[EMAIL]", out)
        self.assertNotIn("mona.example@mail.com", out)


if __name__ == "__main__":
    unittest.main()
