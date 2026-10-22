"""Unit tests for evaluate_accuracy.py — run with:  python -m unittest test_evaluate_accuracy.py"""
import json
import os
import tempfile
import unittest

import evaluate_accuracy as ea

MCQ = {"A": "Facial nerve", "B": "Trigeminal nerve", "C": "Vagus nerve", "D": "Optic nerve"}
YN = {"A": "yes", "B": "no", "C": "maybe"}


class TestNormalize(unittest.TestCase):
    def test_letter_forms(self):
        for raw in ["A", "a", "(A)", "A.", "A)", "A:", " A ", "Answer: A", "The answer is A",
                    "answer is (A).", "Option A", "A) Facial nerve", "A. Facial nerve"]:
            self.assertEqual(ea.normalize_answer(raw, MCQ), "A", raw)

    def test_option_text(self):
        self.assertEqual(ea.normalize_answer("facial nerve", MCQ), "A")
        self.assertEqual(ea.normalize_answer("Yes", YN), "A")
        self.assertEqual(ea.normalize_answer("maybe.", YN), "C")
        self.assertEqual(ea.normalize_answer(True, YN), "A")

    def test_invalid(self):
        for raw in [None, "", "E", "A or B", "I don't know", "A patient with...", ["A", "B"], {}]:
            self.assertIsNone(ea.normalize_answer(raw, MCQ), raw)

    def test_nested_json(self):
        self.assertEqual(ea.normalize_answer({"answer": "B"}, MCQ), "B")
        self.assertEqual(ea.normalize_answer('{"answer": "C", "confidence": 80}', MCQ), "C")

    def test_list_options(self):
        self.assertEqual(ea.normalize_answer("B", ["x", "y"]), "B")
        self.assertIsNone(ea.normalize_answer("C", ["x", "y"]))


class TestStats(unittest.TestCase):
    def test_wilson(self):
        lo, hi = ea.wilson_ci(25, 50)
        self.assertAlmostEqual(lo, 0.3664, places=3)
        self.assertAlmostEqual(hi, 0.6336, places=3)

    def test_mcnemar(self):
        self.assertEqual(ea.mcnemar_exact(0, 0), 1.0)
        # b=10, c=2 -> 2 * P(X<=2 | n=12) = 2 * 79/4096
        self.assertAlmostEqual(ea.mcnemar_exact(10, 2), 2 * 79 / 4096, places=6)
        self.assertEqual(ea.mcnemar_exact(5, 5), 1.0)


class TestEndToEnd(unittest.TestCase):
    def _write(self, path, rows):
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    def test_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            gold = ["A", "B", "C", "D", "A"] * 2
            llm, rag = [], []
            for i, g in enumerate(gold):
                base = {"question": f"Q{i}", "options": MCQ, "correct_answer": g}
                llm.append({**base, "model_answer": "A", "confidence": 90})          # always A
                rag.append({**base, "model_answer": g if i != 0 else "B", "confidence": 70})
            rag[3]["model_answer"] = "not sure"                                       # invalid
            lp, rp = os.path.join(d, "l.jsonl"), os.path.join(d, "r.jsonl")
            self._write(lp, llm)
            self._write(rp, rag)
            with open(lp, "a") as fh:
                fh.write("{broken json\n")
            s = ea.main(["--llm", lp, "--rag", rp, "--out", os.path.join(d, "out"),
                         "--infer-dataset-by-position", "--bootstrap", "500"])
            self.assertEqual(s["overall"]["llm_only"]["correct"], 4)   # gold A at 0,4,5,9
            self.assertEqual(s["overall"]["rag"]["correct"], 8)        # wrong at 0, invalid at 3
            self.assertEqual(s["overall"]["rag"]["n_invalid"], 1)
            p = s["paired"]
            self.assertEqual((p["both_correct"], p["rag_helped"], p["rag_hurt"], p["both_wrong"]), (3, 5, 1, 1))
            self.assertAlmostEqual(p["accuracy_difference"], 0.4)
            self.assertEqual({r["dataset"] for r in s["by_dataset"]}, set(ea.PILOT_DATASET_ORDER))
            for f in ["accuracy_summary.json", "accuracy_overall.csv", "accuracy_by_dataset.csv",
                      "paired_outcomes.csv", "invalid_answers.csv"]:
                self.assertTrue(os.path.exists(os.path.join(d, "out", f)), f)


if __name__ == "__main__":
    unittest.main()
