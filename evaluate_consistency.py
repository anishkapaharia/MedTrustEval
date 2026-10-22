#!/usr/bin/env python3
"""
MedTrustEval evaluation — Dimension 6: Consistency
==================================================
Does the model give the same answer when asked the same question again?

This needs REPEATED runs from the experiment pipeline:

  A. Repeat runs     — the same questions answered R times (R = 5 planned) at temperature 0.7.
                       Save each run as its own file, e.g. logs/repeats/rag_run1.jsonl … rag_run5.jsonl
  B. Shuffled runs   — the same questions with the answer options in a different order.
                       Create the shuffled question file with  make_shuffled_questions.py ,
                       run it through the pipeline, and make sure each output row keeps the
                       "option_map" field (shuffled letter -> original letter).

Metrics (per system)
  Answer agreement     for each question, share of runs that give the most common answer; averaged.
  Fully consistent     share of questions answered identically in every run.
  Flip rate            share of questions where at least one run is correct and at least one is wrong.
  Shuffle stability    share of questions where the shuffled run picks the same answer CONTENT
                       as the original run (after mapping letters back).
  Invalid answers are treated as their own answer ("INVALID").

Usage
    python evaluate_consistency.py \
        --llm-runs "logs/repeats/llm_only_run*.jsonl" \
        --rag-runs "logs/repeats/rag_run*.jsonl" \
        --llm-shuffled logs/shuffled/llm_only_shuffled.jsonl \
        --rag-shuffled logs/shuffled/rag_shuffled.jsonl \
        --llm logs/llm_only_results.jsonl --rag logs/rag_results.jsonl \
        --out results/consistency
"""
from __future__ import annotations

import argparse
import glob
from collections import Counter, defaultdict

from mte_common import get_key, is_correct, load_jsonl, normalize_answer, pct, write_csv, write_json


def repeat_consistency(run_files: list[str]) -> tuple[dict, list[dict]]:
    answers = defaultdict(list)
    correct = defaultdict(list)
    for f in run_files:
        for r in load_jsonl(f):
            k = get_key(r)
            a = normalize_answer(r.get("model_answer"), r.get("options")) or "INVALID"
            answers[k].append(a)
            correct[k].append(is_correct(r)[1])
    per_q = []
    for k, ans in answers.items():
        if len(ans) < 2:
            continue
        top, cnt = Counter(ans).most_common(1)[0]
        per_q.append({"key": k, "n_runs": len(ans), "answers": "".join(a if len(a) == 1 else "?" for a in ans),
                      "majority": top, "agreement": cnt / len(ans),
                      "fully_consistent": cnt == len(ans),
                      "flip": any(correct[k]) and not all(correct[k]),
                      "n_correct": sum(correct[k])})
    if not per_q:
        return {"n_questions": 0, "n_files": len(run_files)}, []
    n = len(per_q)
    return {
        "n_files": len(run_files),
        "n_questions": n,
        "mean_runs_per_question": sum(p["n_runs"] for p in per_q) / n,
        "answer_agreement": sum(p["agreement"] for p in per_q) / n,
        "fully_consistent_rate": sum(p["fully_consistent"] for p in per_q) / n,
        "flip_rate": sum(p["flip"] for p in per_q) / n,
    }, per_q


def shuffle_stability(original_file: str, shuffled_file: str) -> tuple[dict, list[dict]]:
    orig = {get_key(r): r for r in load_jsonl(original_file)}
    rows, missing_map = [], 0
    for r in load_jsonl(shuffled_file):
        k = get_key(r)
        o = orig.get(k)
        if o is None:
            continue
        omap = r.get("option_map")
        if not isinstance(omap, dict):
            missing_map += 1
            continue
        a_shuf = normalize_answer(r.get("model_answer"), r.get("options"))
        a_shuf_orig = omap.get(a_shuf) if a_shuf else None       # map back to original letter
        a_orig = normalize_answer(o.get("model_answer"), o.get("options"))
        rows.append({"key": k, "original_answer": a_orig, "shuffled_answer_mapped": a_shuf_orig,
                     "shuffled_letter": a_shuf, "same_content": a_orig is not None and a_orig == a_shuf_orig,
                     "same_letter": a_orig is not None and a_orig == a_shuf,
                     "orig_correct": is_correct(o)[1], "shuf_correct": is_correct(r)[1]})
    n = len(rows)
    if not n:
        return {"n_questions": 0, "rows_missing_option_map": missing_map}, rows
    return {
        "n_questions": n,
        "shuffle_stability": sum(x["same_content"] for x in rows) / n,
        "same_letter_rate": sum(x["same_letter"] for x in rows) / n,
        "accuracy_original": sum(x["orig_correct"] for x in rows) / n,
        "accuracy_shuffled": sum(x["shuf_correct"] for x in rows) / n,
        "rows_missing_option_map": missing_map,
    }, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="MedTrustEval consistency")
    ap.add_argument("--llm-runs", help='glob, e.g. "logs/repeats/llm_only_run*.jsonl"')
    ap.add_argument("--rag-runs", help='glob, e.g. "logs/repeats/rag_run*.jsonl"')
    ap.add_argument("--llm", default="logs/llm_only_results.jsonl", help="original (unshuffled) run")
    ap.add_argument("--rag", default="logs/rag_results.jsonl", help="original (unshuffled) run")
    ap.add_argument("--llm-shuffled")
    ap.add_argument("--rag-shuffled")
    ap.add_argument("--out", default="results/consistency")
    args = ap.parse_args(argv)

    summary, all_rows = {}, []
    for sysname, pattern, orig, shuf in (("llm_only", args.llm_runs, args.llm, args.llm_shuffled),
                                         ("rag", args.rag_runs, args.rag, args.rag_shuffled)):
        s = {}
        if pattern:
            files = sorted(glob.glob(pattern))
            s["repeat"], rows = repeat_consistency(files)
            all_rows += [{"system": sysname, "test": "repeat", **r} for r in rows]
        if shuf:
            s["shuffle"], rows = shuffle_stability(orig, shuf)
            all_rows += [{"system": sysname, "test": "shuffle", **r} for r in rows]
        summary[sysname] = s

    print("\n=== MedTrustEval · Consistency ===\n")
    print(f"{'metric':<32} {'LLM-only':>10} {'RAG':>10}")
    for label, test, k, is_pct in [("Run files", "repeat", "n_files", False),
                                   ("Questions (repeat test)", "repeat", "n_questions", False),
                                   ("Answer agreement ↑", "repeat", "answer_agreement", True),
                                   ("Fully consistent ↑", "repeat", "fully_consistent_rate", True),
                                   ("Correct/wrong flip rate ↓", "repeat", "flip_rate", True),
                                   ("Questions (shuffle test)", "shuffle", "n_questions", False),
                                   ("Shuffle stability ↑", "shuffle", "shuffle_stability", True),
                                   ("Same letter after shuffle", "shuffle", "same_letter_rate", True),
                                   ("Accuracy, original order", "shuffle", "accuracy_original", True),
                                   ("Accuracy, shuffled order", "shuffle", "accuracy_shuffled", True)]:
        vals = []
        for s in ("llm_only", "rag"):
            v = summary.get(s, {}).get(test, {}).get(k)
            vals.append("—" if v is None else (pct(v) if is_pct else str(v)))
        print(f"{label:<32} {vals[0]:>10} {vals[1]:>10}")
    if not (args.llm_runs or args.rag_runs or args.llm_shuffled or args.rag_shuffled):
        print("\n  Nothing to evaluate yet — this test needs repeated / shuffled runs (see the top of this file).")
    print("\n  Tip: a high 'same letter' rate with low shuffle stability means the model is picking a position, not an answer.")

    write_json(f"{args.out}/consistency_summary.json", summary)
    write_csv(f"{args.out}/consistency_per_question.csv", all_rows)
    print(f"\nSaved to {args.out}/")
    return summary


if __name__ == "__main__":
    main()
