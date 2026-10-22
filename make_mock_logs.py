#!/usr/bin/env python3
"""
Generate FAKE logs in the same format as the real pipeline, so every
evaluation script can be tested before the real results arrive.
The numbers these produce mean nothing.

    python make_mock_logs.py --out mock_logs
"""
from __future__ import annotations

import argparse
import json
import os
import random

DATASETS = ["medqa", "medmcqa", "pubmedqa", "bioasq", "mmlu"]

FACTS = [
    ("Which cranial nerve is affected in Bell's palsy?", ["Facial nerve", "Trigeminal nerve", "Vagus nerve", "Optic nerve"], 0,
     "Bell's palsy is an acute paralysis of the facial nerve, which is cranial nerve seven.",
     "The facial nerve controls the muscles of facial expression."),
    ("Which vitamin deficiency causes scurvy?", ["Vitamin D", "Vitamin C", "Vitamin K", "Vitamin B12"], 1,
     "Scurvy is caused by a deficiency of vitamin C, which is required for collagen synthesis.",
     "Vitamin C deficiency leads to bleeding gums and poor wound healing."),
    ("Which organ produces insulin?", ["Liver", "Kidney", "Pancreas", "Spleen"], 2,
     "Insulin is produced by the beta cells of the pancreas in the islets of Langerhans.",
     "The pancreas has both endocrine and exocrine functions."),
    ("Which electrolyte abnormality causes peaked T waves?", ["Hyponatremia", "Hypocalcemia", "Hypokalemia", "Hyperkalemia"], 3,
     "Hyperkalemia causes peaked T waves on the electrocardiogram.",
     "Severe hyperkalemia can cause cardiac arrest."),
]
YN = [("Does aspirin inhibit cyclooxygenase?", "yes",
       "Aspirin irreversibly inhibits cyclooxygenase enzymes, reducing prostaglandin synthesis."),
      ("Is penicillin effective against viral infections?", "no",
       "Penicillin is an antibiotic and is not effective against viral infections.")]


def make_row(i, rng, system, run=0):
    ds = DATASETS[i // 10 % 5]
    if ds in ("pubmedqa", "bioasq"):
        q, gold_txt, fact = YN[i % len(YN)]
        opts = {"A": "yes", "B": "no", "C": "maybe"} if ds == "pubmedqa" else {"A": "yes", "B": "no"}
        gold = [k for k, v in opts.items() if v == gold_txt][0]
        docs = [{"title": "Pharmacology", "content": fact}] + \
               [{"title": "Misc", "content": f"Unrelated textbook passage number {j} about anatomy."} for j in range(4)]
        good_reason = f"{fact} [1]"
    else:
        q, o, gi, fact, fact2 = FACTS[i % len(FACTS)]
        opts = dict(zip("ABCD", o))
        gold = "ABCD"[gi]
        docs = [{"title": "Textbook", "content": fact}, {"title": "Textbook", "content": fact2}] + \
               [{"title": "Misc", "content": f"Unrelated textbook passage number {j} about histology."} for j in range(3)]
        good_reason = f"{fact} [1] {fact2} [2]"
    q = f"{q} (item {i})"
    if system == "llm_only":
        ans = "A" if rng.random() < 0.7 else rng.choice(list(opts))           # position bias
        reasoning = good_reason.replace(" [1]", "").replace(" [2]", "")
        if ans != gold:
            reasoning = "The answer is not related to " + opts[gold].lower() + ". " + reasoning.split(". ")[0] + " is not true."
        conf = rng.choice([85, 90, 95])
    else:
        ans = gold if rng.random() < 0.75 else rng.choice(list(opts))
        reasoning = good_reason if rng.random() > 0.22 else good_reason.replace(" [2]", "").replace("[1]", "[7]")
        conf = rng.choice([60, 70, 80, 90]) if ans == gold else rng.choice([40, 60, 80])
    row = {"id": f"{ds}-{i}", "dataset": ds, "question": q, "options": opts, "correct_answer": gold,
           "model_answer": ans if rng.random() > 0.04 else "I am not sure",
           "confidence": conf, "reasoning": reasoning, "temperature": 0.7, "model": "llama3:8b",
           "system": system, "run": run, "timestamp": "2026-08-19T12:00:00"}
    if system == "rag":
        row["retrieved_docs"] = docs
        row["citations_valid"] = "[7]" not in reasoning
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mock_logs")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args(argv)
    os.makedirs(f"{args.out}/repeats", exist_ok=True)
    for system in ("llm_only", "rag"):
        rng = random.Random(1 if system == "llm_only" else 2)
        with open(f"{args.out}/{system}_results.jsonl", "w") as fh:
            for i in range(args.n):
                fh.write(json.dumps(make_row(i, rng, system)) + "\n")
        for run in range(1, args.runs + 1):
            rng = random.Random(100 * run + len(system))
            with open(f"{args.out}/repeats/{system}_run{run}.jsonl", "w") as fh:
                for i in range(args.n):
                    fh.write(json.dumps(make_row(i, rng, system, run)) + "\n")
    # questions file (what Justin's pipeline reads)
    with open(f"{args.out}/questions.jsonl", "w") as fh:
        rng = random.Random(0)
        for i in range(args.n):
            r = make_row(i, rng, "rag")
            fh.write(json.dumps({k: r[k] for k in ("id", "dataset", "question", "options", "correct_answer")}) + "\n")
    print(f"Mock logs written to {args.out}/")


if __name__ == "__main__":
    main()
