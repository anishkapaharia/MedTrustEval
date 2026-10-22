"""Tests for the hallucination/grounding, calibration, consistency and review scripts.
Run with:  python -m unittest discover -p "test_*.py"
"""
import json
import os
import tempfile
import unittest

import evaluate_calibration as cal
import evaluate_consistency as con
import evaluate_hallucination_grounding as hg
import make_mock_logs
import make_shuffled_questions as msq
import mte_common as mc
import review_sheets as rs
import run_all
from nli_judge import LexicalJudge


class FakeJudge:
    """Entails when the hypothesis text appears inside the premise; contradicts when marked 'FALSE'."""
    model_name = "fake"

    def score(self, pairs):
        out = []
        for p, h in pairs:
            core = h.rstrip(".")
            if "FALSE" in h:
                out.append({"entailment": 0.0, "contradiction": 0.9, "neutral": 0.1})
            elif core and core in p:
                out.append({"entailment": 0.95, "contradiction": 0.0, "neutral": 0.05})
            else:
                out.append({"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0})
        return out


class TestCommon(unittest.TestCase):
    def test_split_and_citations(self):
        s = mc.split_sentences("The facial nerve is CN VII [1]. It controls expression. [2][3] See e.g. Fig 2.")
        self.assertEqual(len(s), 3)
        self.assertEqual(mc.sentence_citations(s[0]), [1])
        self.assertEqual(mc.sentence_citations(s[1]), [2, 3])
        self.assertEqual(mc.sentence_citations("x [1, 4]"), [1, 4])
        self.assertEqual(mc.strip_citations("A [1] b [2,3]."), "A b.")

    def test_confidence(self):
        self.assertEqual(mc.get_confidence({"confidence": 85}), 0.85)
        self.assertEqual(mc.get_confidence({"confidence": "70%"}), 0.70)
        self.assertEqual(mc.get_confidence({"confidence": 0.3}), 0.3)
        self.assertIsNone(mc.get_confidence({"confidence": "high"}))
        self.assertIsNone(mc.get_confidence({"confidence": 250}))

    def test_docs(self):
        self.assertEqual(mc.get_docs({"retrieved_docs": [{"title": "T", "content": "C"}, "raw"]}), ["T. C", "raw"])
        self.assertEqual(mc.get_docs({"retrieved_docs": {"2": "b", "1": "a"}}), ["a", "b"])


class TestHallucination(unittest.TestCase):
    def rows(self):
        docs = ["Bell palsy affects the facial nerve", "The facial nerve controls expression", "x", "y", "z"]
        opts = {"A": "Facial nerve", "B": "Vagus nerve"}
        base = {"id": "q1", "question": "Which nerve?", "options": opts, "correct_answer": "A"}
        rag = dict(base, model_answer="A", confidence=90, retrieved_docs=docs,
                   reasoning="Bell palsy affects the facial nerve [1]. The facial nerve controls expression [1]. "
                             "Something unrelated here [9].")
        llm = dict(base, model_answer="B", confidence=95,
                   reasoning="Bell palsy affects the facial nerve. The vagus nerve is FALSE here.")
        return [llm], [rag]

    def test_labels_and_citations(self):
        llm, rag = self.rows()
        st, rp, ci, hits, miss = hg.evaluate(llm, rag, FakeJudge())
        s = hg.summarize(st, rp, ci, hits)
        self.assertEqual(miss, {})
        self.assertAlmostEqual(s["llm_only"]["contradicted_statement_rate"], 0.5)
        self.assertEqual(s["llm_only"]["response_hallucination_rate"], 1.0)
        self.assertEqual(s["llm_only"]["confident_error_rate"], 1.0)
        self.assertAlmostEqual(s["rag"]["faithfulness"], 2 / 3)
        self.assertEqual(s["rag"]["citation_validity"], 0.0)        # [9] is out of range
        self.assertAlmostEqual(s["rag"]["citation_recall"], 1 / 3)  # only statement 1 is supported by what it cites
        # statement 2 ("The facial nerve controls expression") cites [1], which doesn't contain it
        # -> citation 2 irrelevant; statement 1 citation relevant
        self.assertAlmostEqual(s["rag"]["citation_precision"], 0.5)
        self.assertEqual(s["rag"]["evidence_hit_rate"], 0.0)       # hypothesis text is not literally in a doc

    def test_citation_recall_corrected(self):
        llm, rag = self.rows()
        rag[0]["reasoning"] = "Bell palsy affects the facial nerve [1]. The facial nerve controls expression [2]."
        st, rp, ci, hits, _ = hg.evaluate(llm, rag, FakeJudge())
        s = hg.summarize(st, rp, ci, hits)
        self.assertEqual(s["rag"]["citation_validity"], 1.0)
        self.assertEqual(s["rag"]["citation_recall"], 1.0)
        self.assertEqual(s["rag"]["citation_precision"], 1.0)
        self.assertEqual(s["rag"]["fully_grounded_rate"], 1.0)

    def test_missing_evidence(self):
        llm, rag = self.rows()
        rag[0]["retrieved_docs"] = []
        _, _, _, _, miss = hg.evaluate(llm, rag, FakeJudge())
        self.assertEqual(miss, {"llm_only": 1, "rag": 1})


class TestCalibration(unittest.TestCase):
    def test_perfect_and_bad(self):
        perfect = [(1.0, True)] * 10 + [(0.0, False)] * 10
        r = cal.calibration(perfect)
        self.assertAlmostEqual(r["ece"], 0.0)
        self.assertAlmostEqual(r["brier"], 0.0)
        bad = [(0.9, False)] * 10
        r = cal.calibration(bad)
        self.assertAlmostEqual(r["ece"], 0.9)
        self.assertAlmostEqual(r["brier"], 0.81)
        self.assertEqual(r["confident_error_rate"], 1.0)

    def test_ece_mixed(self):
        # bin 0.7-0.8: conf .75 x4, 3 correct -> |0.75-0.75|=0 ; bin 0.9-1.0: conf .95 x4, 2 correct -> .45
        pairs = [(0.75, True)] * 3 + [(0.75, False)] + [(0.95, True)] * 2 + [(0.95, False)] * 2
        self.assertAlmostEqual(cal.calibration(pairs)["ece"], 0.5 * 0.45)


class TestConsistencyAndShuffle(unittest.TestCase):
    def test_shuffle_moves_gold(self):
        import random
        row = {"id": "x", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct_answer": "A"}
        for seed in range(20):
            new = msq.shuffle_row(row, random.Random(seed), seed)
            self.assertNotEqual(new["correct_answer"], "A")
            self.assertEqual(new["options"][new["correct_answer"]], "a")
            self.assertEqual(new["option_map"][new["correct_answer"]], "A")

    def test_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            for run, ans in enumerate(["A", "A", "B"], 1):
                with open(f"{d}/r{run}.jsonl", "w") as fh:
                    fh.write(json.dumps({"id": "q", "options": {"A": "x", "B": "y"}, "correct_answer": "A",
                                         "model_answer": ans}) + "\n")
            s, rows = con.repeat_consistency(sorted(__import__("glob").glob(f"{d}/r*.jsonl")))
            self.assertAlmostEqual(s["answer_agreement"], 2 / 3)
            self.assertEqual(s["fully_consistent_rate"], 0.0)
            self.assertEqual(s["flip_rate"], 1.0)


class TestKappa(unittest.TestCase):
    def test_kappa(self):
        self.assertEqual(rs.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)
        self.assertAlmostEqual(rs.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]), 0.0)
        self.assertIsNone(rs.cohen_kappa(["", None], ["", ""]))


class TestEndToEnd(unittest.TestCase):
    def test_run_all_on_mock(self):
        with tempfile.TemporaryDirectory() as d:
            cwd = os.getcwd()
            make_mock_logs.main(["--out", f"{d}/logs", "--n", "20", "--runs", "3"])
            run_all.main(["--llm", f"{d}/logs/llm_only_results.jsonl", "--rag", f"{d}/logs/rag_results.jsonl",
                          "--out", f"{d}/results", "--judge", "lexical",
                          "--rag-runs", f"{d}/logs/repeats/rag_run*.jsonl"])
            text = open(f"{d}/results/RESULTS_TABLES.md").read()
            self.assertIn("Table 5", text)
            self.assertIn("WARNING", text)
            self.assertEqual(os.getcwd(), cwd)



class TestJustinSchema(unittest.TestCase):
    """A record in exactly the format described in Justin's 'RAG Run — Answers & Retrieved Sources' doc."""
    REC = {
        "dataset": "medqa", "dataset_index": "test",
        "question": "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral",
        "options": {"A": "paralysis of the facial muscles.",
                    "B": "paralysis of the facial muscles and loss of taste.",
                    "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
                    "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."},
        "correct_answer": "A", "model_answer": "A", "confidence": 75,
        "reasoning": "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause "
                     "paralysis of the facial muscles [1].",
        "citations": [{"sentence": "A lesion ... [1].", "citation": 1}],
        "retrieved_docs": [{"id": "d1", "title": "InternalMed_Harrison",
                            "content": "A lesion causing compression of the facial nerve at the stylomastoid "
                                       "foramen will cause paralysis of the facial muscles"},
                           {"id": "d2", "title": "Anatomy_Gray", "content": "The facial nerve [VII] ..."}],
        "retrieval_scores": [0.91, 0.88], "citations_valid": True, "temperature": 0.7,
        "model": "llama3:8b", "system": "rag", "timestamp": "2026-08-26T20:00:00",
    }

    def test_schema(self):
        rag = dict(self.REC)
        llm = {k: v for k, v in self.REC.items()
               if k not in ("citations", "retrieved_docs", "retrieval_scores", "citations_valid")}
        llm.update(system="llm_only", model_answer="D", confidence=90, reasoning="It causes everything.")
        self.assertEqual(mc.get_key(rag), mc.get_key(llm))
        self.assertEqual(mc.get_dataset(rag), "medqa")
        self.assertEqual(mc.is_correct(rag), (True, True))
        self.assertEqual(len(mc.get_docs(rag)), 2)
        st, rp, ci, hits, miss = hg.evaluate([llm], [rag], FakeJudge())
        s = hg.summarize(st, rp, ci, hits)
        self.assertEqual(s["rag"]["faithfulness"], 1.0)
        self.assertEqual(s["rag"]["citation_validity"], 1.0)
        self.assertEqual(s["rag"]["logged_citation_validity"], 1.0)
        self.assertEqual(s["llm_only"]["confident_error_rate"], 1.0)


class TestVerbatimCopy(unittest.TestCase):
    def test_copy_flag(self):
        docs = ["In fact, growing up on a farm with domestic animals appears to protect against developing asthma."]
        self.assertTrue(hg.is_verbatim_copy("Growing up on a farm with domestic animals appears to protect against developing asthma", docs))
        self.assertFalse(hg.is_verbatim_copy("Farm life protects children from asthma according to studies", docs))
        self.assertFalse(hg.is_verbatim_copy("farm animals", docs))   # too short to count

    def test_summary_has_copy_metrics(self):
        docs = ["Bell palsy affects the facial nerve and causes weakness of the face", "x", "y", "z", "w"]
        base = {"id": "q", "question": "Q?", "options": {"A": "a", "B": "b"}, "correct_answer": "A"}
        rag = [dict(base, model_answer="A", confidence=70, retrieved_docs=docs,
                    reasoning="Bell palsy affects the facial nerve and causes weakness of the face [1]. Unrelated claim here [2].")]
        llm = [dict(base, model_answer="A", confidence=70, reasoning="Something.")]
        st, rp, ci, hits, _ = hg.evaluate(llm, rag, FakeJudge())
        s = hg.summarize(st, rp, ci, hits)["rag"]
        self.assertAlmostEqual(s["verbatim_copy_rate"], 0.5)
        self.assertAlmostEqual(s["faithfulness"], 0.5)
        self.assertAlmostEqual(s["faithfulness_excluding_copies"], 0.0)


class TestPlaceholder(unittest.TestCase):
    def test_placeholder_removed(self):
        self.assertEqual(mc.get_reasoning({"reasoning": "One sentence explanation"}), "")
        self.assertEqual(mc.get_reasoning({"reasoning": "One sentence explanation: Aspirin inhibits COX."}), "Aspirin inhibits COX.")
        self.assertEqual(mc.get_reasoning({"reasoning": "Aspirin inhibits COX."}), "Aspirin inhibits COX.")


class TestKeyMatching(unittest.TestCase):
    def test_id_and_no_id_match(self):
        old = {"dataset": "medqa", "question": "Which nerve?  "}
        new = {"dataset": "medqa", "question": "Which nerve?", "id": "medqa-0"}
        self.assertEqual(mc.get_key(old), mc.get_key(new))
        self.assertNotEqual(mc.get_key(old), mc.get_key({"dataset": "mmlu", "question": "Which nerve?"}))


class TestRunExperimentsWithFakeOllama(unittest.TestCase):
    """End-to-end test of run_experiments.py against a fake Ollama server (no model needed)."""

    def test_end_to_end(self):
        import http.server
        import threading
        import run_experiments as rx
        import evaluate_consistency as ec

        class Fake(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                rag = "Documents:" in body["prompt"]
                assert body["options"]["temperature"] == 0.7 and body["format"] == "json"
                assert body["options"].get("num_thread") in (None, 2)
                assert body["options"].get("num_ctx") in (None, 4096)
                reasoning = "The facial nerve is affected [1]. It controls expression [2]." if rag else \
                    "The facial nerve is affected. It controls expression."
                out = json.dumps({"answer": "A", "confidence": 80, "reasoning": reasoning})
                data = json.dumps({"response": out}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{srv.server_address[1]}/api/generate"
        with tempfile.TemporaryDirectory() as d:
            src = f"{d}/rag.jsonl"
            with open(src, "w") as fh:
                for i in range(4):
                    fh.write(json.dumps({"dataset": "mmlu", "dataset_index": i, "question": f"Q{i}?",
                                         "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct_answer": "B",
                                         "retrieved_docs": [{"title": "T", "content": "c"}] * 5}) + "\n")
            rx.main(["--source", src, "--out-dir", f"{d}/v2", "--run", "1", "--url", url, "--threads", "2",
                     "--num-ctx", "4096", "--keep-alive", "0"])
            rx.main(["--source", src, "--out-dir", f"{d}/v2", "--run", "1", "--url", url])   # resume: adds nothing
            rx.main(["--source", src, "--out-dir", f"{d}/v2/shuffled", "--shuffle", "--url", url])
            rag = [json.loads(l) for l in open(f"{d}/v2/rag_run1.jsonl")]
            llm = [json.loads(l) for l in open(f"{d}/v2/llm_only_run1.jsonl")]
            self.assertEqual((len(rag), len(llm)), (4, 4))                 # resume didn't duplicate
            self.assertTrue(all(r["citations_valid"] for r in rag))
            self.assertEqual(rag[0]["retrieved_docs"][0]["title"], "T")    # passages reused
            sh = [json.loads(l) for l in open(f"{d}/v2/shuffled/rag_shuffled.jsonl")]
            self.assertTrue(all("option_map" in r for r in sh))
            self.assertTrue(all(r["correct_answer"] != "B" for r in sh))   # gold moved
            s, _ = ec.shuffle_stability(f"{d}/v2/rag_run1.jsonl", f"{d}/v2/shuffled/rag_shuffled.jsonl")
            self.assertEqual(s["n_questions"], 4)
            self.assertEqual(s["same_letter_rate"], 1.0)                   # always-"A" model detected
        srv.shutdown()


if __name__ == "__main__":
    unittest.main()
