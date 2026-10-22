#!/usr/bin/env python3
"""
Manual review sheets for MedTrustEval (Day 9 "review examples together").

  make   → creates three CSV sheets to fill in (open them in Excel or Google Sheets)
  score  → reads the filled-in sheets and computes Precision@5, Cohen's kappa, and
           how often the automatic NLI labels agree with the human labels

Sheets
  review_examples.csv      Paired examples in each outcome group (RAG helped / RAG hurt / both wrong /
                           both correct), with both answers, both explanations and the retrieved passages.
                           Fill in: problem_source (retrieval / evidence_use / citation / question / other), notes
  retrieval_relevance.csv  One row per (question, retrieved passage).
                           Fill in: rater1_relevant, rater2_relevant  (1 = relevant, 0 = not relevant)
  statement_labels.csv     A random sample of explanation statements with the automatic label hidden.
                           Fill in: rater1_label, rater2_label  (S = supported, C = contradicted, N = not addressed)

Usage
    python review_sheets.py make  --llm logs/llm_only_results.jsonl --rag logs/rag_results.jsonl \
        --paired results/accuracy/paired_outcomes.csv --statements results/hallucination/statements.csv \
        --out review --n-questions 10 --n-statements 60
    python review_sheets.py score --out review
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from collections import Counter, defaultdict

from mte_common import get_docs, get_key, get_reasoning, load_jsonl, write_csv, write_json

LABEL_MAP = {"s": "supported", "c": "contradicted", "n": "not_addressed",
             "supported": "supported", "contradicted": "contradicted", "not_addressed": "not_addressed",
             "not addressed": "not_addressed"}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def cohen_kappa(a: list, b: list) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x not in (None, "") and y not in (None, "")]
    n = len(pairs)
    if n == 0:
        return None
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def make(args):
    rng = random.Random(args.seed)
    llm = {get_key(r): r for r in load_jsonl(args.llm)}
    rag = {get_key(r): r for r in load_jsonl(args.rag)}
    os.makedirs(args.out, exist_ok=True)

    # 1. paired examples
    groups = defaultdict(list)
    if args.paired and os.path.exists(args.paired):
        for p in read_csv(args.paired):
            groups[p["outcome"]].append(p["key"])
    ex_rows = []
    for outcome in ("rag_helped", "rag_hurt", "both_wrong", "both_correct"):
        keys = groups.get(outcome, [])
        for k in rng.sample(keys, min(args.per_group, len(keys))):
            l, r = llm.get(k, {}), rag.get(k, {})
            docs = get_docs(r)
            ex_rows.append({
                "outcome": outcome, "key": k, "question": r.get("question") or l.get("question"),
                "options": r.get("options") or l.get("options"),
                "correct_answer": r.get("correct_answer") or l.get("correct_answer"),
                "llm_answer": l.get("model_answer"), "llm_confidence": l.get("confidence"),
                "llm_reasoning": get_reasoning(l),
                "rag_answer": r.get("model_answer"), "rag_confidence": r.get("confidence"),
                "rag_reasoning": get_reasoning(r),
                **{f"doc{i + 1}": d[:1500] for i, d in enumerate(docs[:5])},
                "problem_source": "", "notes": "",
            })
    write_csv(f"{args.out}/review_examples.csv", ex_rows)

    # 2. retrieval relevance
    keys = [k for k in rag if get_docs(rag[k])]
    rel_rows = []
    for k in rng.sample(keys, min(args.n_questions, len(keys))):
        r = rag[k]
        for i, d in enumerate(get_docs(r)[:5], 1):
            rel_rows.append({"key": k, "question": r.get("question"), "doc_rank": i, "passage": d[:1500],
                             "rater1_relevant": "", "rater2_relevant": ""})
    write_csv(f"{args.out}/retrieval_relevance.csv", rel_rows)

    # 3. statement labels (auto label kept in a separate file so raters don't see it)
    st_rows, key_rows = [], []
    if args.statements and os.path.exists(args.statements):
        stmts = read_csv(args.statements)
        for j, s in enumerate(rng.sample(stmts, min(args.n_statements, len(stmts))), 1):
            src = rag.get(s["key"], {})
            best = s.get("best_doc")
            docs = get_docs(src)
            st_rows.append({"item": j, "system": s["system"], "key": s["key"], "statement": s["statement"],
                            **{f"doc{i + 1}": d[:1200] for i, d in enumerate(docs[:5])},
                            "rater1_label": "", "rater2_label": ""})
            key_rows.append({"item": j, "auto_label": s["label"], "best_doc": best})
    write_csv(f"{args.out}/statement_labels.csv", st_rows)
    write_csv(f"{args.out}/_statement_auto_labels_DO_NOT_OPEN_BEFORE_RATING.csv", key_rows)

    print(f"Created in {args.out}/: review_examples.csv ({len(ex_rows)} rows), "
          f"retrieval_relevance.csv ({len(rel_rows)} rows), statement_labels.csv ({len(st_rows)} rows)")
    if not ex_rows:
        print("  (no paired examples — run evaluate_accuracy.py first and pass --paired)")
    if not st_rows:
        print("  (no statements — run evaluate_hallucination_grounding.py first and pass --statements)")


def _bin(v):
    v = str(v).strip().lower()
    return {"1": 1, "y": 1, "yes": 1, "0": 0, "n": 0, "no": 0}.get(v)


def score(args):
    out = {}
    p = f"{args.out}/retrieval_relevance.csv"
    if os.path.exists(p):
        rows = read_csv(p)
        r1 = [_bin(r["rater1_relevant"]) for r in rows]
        r2 = [_bin(r["rater2_relevant"]) for r in rows]
        per_q = defaultdict(list)
        for r, a, b in zip(rows, r1, r2):
            vals = [x for x in (a, b) if x is not None]
            if vals:
                per_q[r["key"]].append(sum(vals) / len(vals))  # average of raters
        out["retrieval"] = {
            "n_questions": len(per_q),
            "precision_at_5": (sum(sum(v) / len(v) for v in per_q.values()) / len(per_q)) if per_q else None,
            "rater1_precision": (sum(x for x in r1 if x is not None) / max(1, sum(x is not None for x in r1))),
            "rater2_precision": (sum(x for x in r2 if x is not None) / max(1, sum(x is not None for x in r2))),
            "cohen_kappa": cohen_kappa(r1, r2),
        }
    p = f"{args.out}/statement_labels.csv"
    k = f"{args.out}/_statement_auto_labels_DO_NOT_OPEN_BEFORE_RATING.csv"
    if os.path.exists(p):
        rows = read_csv(p)
        auto = {r["item"]: r["auto_label"] for r in read_csv(k)} if os.path.exists(k) else {}
        r1 = [LABEL_MAP.get(r["rater1_label"].strip().lower()) for r in rows]
        r2 = [LABEL_MAP.get(r["rater2_label"].strip().lower()) for r in rows]
        agreed = [(a, auto.get(r["item"])) for r, a, b in zip(rows, r1, r2) if a and a == b]
        out["statements"] = {
            "n_items": len(rows),
            "n_labelled_by_both": sum(1 for a, b in zip(r1, r2) if a and b),
            "cohen_kappa_raters": cohen_kappa(r1, r2),
            "n_rater_agreed": len(agreed),
            "auto_vs_human_accuracy": (sum(h == a for h, a in agreed) / len(agreed)) if agreed else None,
            "auto_vs_human_kappa": cohen_kappa([h for h, _ in agreed], [a for _, a in agreed]) if agreed else None,
            "confusion_human_to_auto": {f"{h}->{a}": c for (h, a), c in Counter(agreed).items()},
        }
    write_json(f"{args.out}/manual_review_scores.json", out)
    print(json_pretty(out))


def json_pretty(o):
    import json
    return json.dumps(o, indent=2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--llm", default="logs/llm_only_results.jsonl")
    m.add_argument("--rag", default="logs/rag_results.jsonl")
    m.add_argument("--paired", default="results/accuracy/paired_outcomes.csv")
    m.add_argument("--statements", default="results/hallucination/statements.csv")
    m.add_argument("--out", default="review")
    m.add_argument("--per-group", type=int, default=5)
    m.add_argument("--n-questions", type=int, default=10)
    m.add_argument("--n-statements", type=int, default=60)
    m.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("score")
    s.add_argument("--out", default="review")
    args = ap.parse_args(argv)
    (make if args.cmd == "make" else score)(args)


if __name__ == "__main__":
    main()
