#!/usr/bin/env python3
"""
MedTrustEval evaluation — Dimension 1: Accuracy
===============================================

Compares LLM-only vs. LLM+RAG answers on the same MIRAGE questions.

Reads the two JSONL logs produced by the experiment pipeline
(one JSON object per line, with at least `correct_answer` and `model_answer`):

    logs/llm_only_results.jsonl
    logs/rag_results.jsonl

and reports:

  * Overall accuracy per system, with a 95% Wilson confidence interval
  * Invalid-answer rate (missing / unparseable answers — counted as wrong)
  * Accuracy per source dataset (medqa, medmcqa, pubmedqa, bioasq, mmlu)
  * Answer-letter distribution (to expose position bias, e.g. always "A")
  * Paired comparison on shared questions:
        both correct / RAG helped / RAG hurt / both wrong,
        exact McNemar test, and a bootstrap 95% CI for the accuracy difference
  * A per-question CSV listing where RAG helped or hurt (for error analysis)

Only the Python standard library is used, so it runs anywhere.

Usage
-----
    python evaluate_accuracy.py \
        --llm logs/llm_only_results.jsonl \
        --rag logs/rag_results.jsonl \
        --out results/accuracy

    # If the log rows have no dataset field, and the file is the
    # 50-question pilot (10 each, in this order), infer it from position:
    python evaluate_accuracy.py ... --infer-dataset-by-position

Scoring rules (documented so the paper's Methods section can cite them)
-----------------------------------------------------------------------
  1. Gold and predicted answers are normalized to an option letter (A, B, C, ...).
  2. A prediction is accepted if it is a bare letter ("B", "(B)", "B.", "B) text",
     "Answer: B", "option B"), or if it exactly matches one option's text
     (case-insensitive), e.g. "yes" -> the letter whose option is "yes".
  3. Anything else (empty, refusal, several letters, text matching no option)
     is INVALID and scored as incorrect. Invalid answers are also reported
     separately so they are not hidden inside the accuracy number.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, OrderedDict, defaultdict

# Order of datasets in the 50-question pilot (first 10 of each).
PILOT_DATASET_ORDER = ["medqa", "medmcqa", "pubmedqa", "bioasq", "mmlu"]
KNOWN_DATASETS = ["medqa", "medmcqa", "pubmedqa", "bioasq", "mmlu"]
DATASET_FIELDS = ["dataset", "source", "task", "subset", "benchmark"]
ID_FIELDS = ["id", "question_id", "qid", "uid"]

LETTERS = "ABCDEFGHIJ"

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file; skip (and report) malformed lines instead of crashing."""
    rows, bad = [], 0
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                print(f"  [warn] {path}:{line_no} is not valid JSON — skipped", file=sys.stderr)
                continue
            if isinstance(obj, dict):
                obj["_line"] = line_no
                rows.append(obj)
    if bad:
        print(f"  [warn] {bad} malformed line(s) skipped in {path}", file=sys.stderr)
    return rows


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def options_as_dict(options) -> "OrderedDict[str, str]":
    """Accept options as {"A": "..."} or ["...", "..."] or None."""
    if isinstance(options, dict):
        return OrderedDict((str(k).strip().upper(), str(v)) for k, v in options.items())
    if isinstance(options, (list, tuple)):
        return OrderedDict((LETTERS[i], str(v)) for i, v in enumerate(options))
    return OrderedDict()


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


_LETTER_PATTERNS = [
    r"^\(?([A-J])\)?[.:)]?$",                                  # A  (A)  A.  A)  A:
    r"^\(?([A-J])\)?[.:)]\s+\S",                               # "B) text", "B. text"
    r"^(?:the\s+)?(?:final\s+)?(?:answer|option|choice)\s*(?:is)?\s*[:\-]?\s*\(?([A-J])\)?(?:[.:)\s]|$)",
]


def normalize_answer(raw, options=None) -> str | None:
    """
    Map a raw answer to an option letter, or return None if it can't be
    mapped unambiguously. See module docstring for the rules.
    """
    if raw is None:
        return None
    # The model sometimes returns the whole JSON object, or a nested dict.
    if isinstance(raw, dict):
        for key in ("answer", "answer_choice", "choice", "prediction"):
            if key in raw:
                return normalize_answer(raw[key], options)
        return None
    if isinstance(raw, (list, tuple)):
        return normalize_answer(raw[0], options) if len(raw) == 1 else None
    if isinstance(raw, bool):
        raw = "yes" if raw else "no"
    text = str(raw).strip()
    if not text:
        return None
    # A JSON string inside the string
    if text.startswith("{"):
        try:
            return normalize_answer(json.loads(text), options)
        except json.JSONDecodeError:
            pass

    opts = options_as_dict(options)
    valid_letters = set(opts) if opts else set(LETTERS)

    upper = text.upper()
    for pat in _LETTER_PATTERNS:
        m = re.match(pat, upper, flags=re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            return letter if letter in valid_letters else None

    # Exact match against option text (e.g. "yes", "no", "maybe", full option)
    if opts:
        cleaned = _clean(text)
        matches = [k for k, v in opts.items() if _clean(v) == cleaned]
        if len(matches) == 1:
            return matches[0]
        # "B: Facial nerve" style — letter + the option's own text
        m = re.match(r"^\(?([A-J])\)?\s*[.:)\-]\s*(.+)$", text, flags=re.IGNORECASE)
        if m and m.group(1).upper() in opts and _clean(m.group(2)) == _clean(opts[m.group(1).upper()]):
            return m.group(1).upper()

    return None


# --------------------------------------------------------------------------- #
# Per-row helpers
# --------------------------------------------------------------------------- #


def get_dataset(row: dict) -> str | None:
    for f in DATASET_FIELDS:
        if row.get(f):
            return str(row[f]).strip().lower()
    for f in ID_FIELDS:
        if row.get(f):
            rid = str(row[f]).lower()
            for d in KNOWN_DATASETS:
                if rid.startswith(d):
                    return d
    return None


def get_key(row: dict) -> str:
    """
    Stable key used to pair the same question across files.
    Uses dataset + question text whenever the question is present, so files with and
    without an "id" field (e.g. old pilot logs vs. new runs) still match. Falls back to "id".
    """
    q = _clean(str(row.get("question", "")))
    if q:
        ds = get_dataset(row) or ""
        return f"q:{ds}:" + hashlib.sha1(q.encode()).hexdigest()[:16]
    for f in ID_FIELDS:
        if row.get(f) not in (None, ""):
            return f"id:{row[f]}"
    return "q::empty"


def score_rows(rows: list[dict], system: str, infer_by_position: bool) -> list[dict]:
    scored = []
    for i, r in enumerate(rows):
        opts = r.get("options")
        gold = normalize_answer(r.get("correct_answer", r.get("answer")), opts)
        pred = normalize_answer(r.get("model_answer"), opts)
        dataset = get_dataset(r)
        if dataset is None and infer_by_position and len(rows) % len(PILOT_DATASET_ORDER) == 0:
            block = len(rows) // len(PILOT_DATASET_ORDER)
            dataset = PILOT_DATASET_ORDER[i // block]
        scored.append({
            "key": get_key(r),
            "system": system,
            "dataset": dataset or "unknown",
            "question": str(r.get("question", ""))[:200],
            "gold": gold,
            "pred": pred,
            "raw_answer": r.get("model_answer"),
            "valid": pred is not None,
            "gold_ok": gold is not None,
            "correct": pred is not None and gold is not None and pred == gold,
            "confidence": r.get("confidence"),
            "line": r.get("_line"),
        })
    return scored


def dedupe(scored: list[dict]) -> list[dict]:
    """If a question appears more than once (re-runs), keep the last row."""
    seen = OrderedDict()
    for s in scored:
        seen[s["key"]] = s
    if len(seen) < len(scored):
        print(f"  [warn] {len(scored) - len(seen)} duplicate question(s) in "
              f"{scored[0]['system']} — keeping the last occurrence", file=sys.stderr)
    return list(seen.values())


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (better than normal approx for small n)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """
    Exact two-sided McNemar test. b = RAG helped (LLM wrong, RAG right),
    c = RAG hurt (LLM right, RAG wrong). Under H0 the discordant pairs
    are Binomial(b+c, 0.5).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_diff_ci(pairs: list[tuple[int, int]], iters: int = 10000, seed: int = 42):
    """Paired bootstrap 95% CI for accuracy(RAG) - accuracy(LLM)."""
    if not pairs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(pairs)
    diffs = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            llm, rag = pairs[rng.randrange(n)]
            s += rag - llm
        diffs.append(s / n)
    diffs.sort()
    return (diffs[int(0.025 * iters)], diffs[int(0.975 * iters) - 1])


def summarize(scored: list[dict]) -> dict:
    n = len(scored)
    k = sum(s["correct"] for s in scored)
    valid = [s for s in scored if s["valid"]]
    lo, hi = wilson_ci(k, n)
    return {
        "n": n,
        "correct": k,
        "accuracy": k / n if n else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
        "n_invalid": n - len(valid),
        "invalid_rate": (n - len(valid)) / n if n else float("nan"),
        "accuracy_on_valid": (sum(s["correct"] for s in valid) / len(valid)) if valid else float("nan"),
        "n_gold_unparsed": sum(not s["gold_ok"] for s in scored),
        "pred_distribution": dict(sorted(Counter(s["pred"] or "INVALID" for s in scored).items())),
        "gold_distribution": dict(sorted(Counter(s["gold"] or "?" for s in scored).items())),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def pct(x: float) -> str:
    return "  n/a " if x != x else f"{100 * x:5.1f}%"


def write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description="MedTrustEval accuracy evaluation (LLM-only vs RAG)")
    ap.add_argument("--llm", default="logs/llm_only_results.jsonl", help="LLM-only JSONL log")
    ap.add_argument("--rag", default="logs/rag_results.jsonl", help="RAG JSONL log")
    ap.add_argument("--out", default="results/accuracy", help="output directory")
    ap.add_argument("--infer-dataset-by-position", action="store_true",
                    help="assume equal blocks in order " + ", ".join(PILOT_DATASET_ORDER))
    ap.add_argument("--bootstrap", type=int, default=10000, help="bootstrap iterations")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    systems = OrderedDict()
    for name, path in (("llm_only", args.llm), ("rag", args.rag)):
        if not os.path.exists(path):
            sys.exit(f"File not found: {path}")
        systems[name] = dedupe(score_rows(load_jsonl(path), name, args.infer_dataset_by_position))
        if not systems[name]:
            sys.exit(f"No rows loaded from {path}")

    # ---- 1. Overall ------------------------------------------------------- #
    overall = {name: summarize(rows) for name, rows in systems.items()}
    print("\n=== MedTrustEval · Accuracy ===\n")
    print(f"{'system':<10} {'n':>4} {'correct':>8} {'accuracy':>9}   {'95% CI':<17} {'invalid':>8} {'acc(valid)':>11}")
    for name, s in overall.items():
        ci = f"[{pct(s['ci_low']).strip()}, {pct(s['ci_high']).strip()}]"
        print(f"{name:<10} {s['n']:>4} {s['correct']:>8} {pct(s['accuracy']):>9}   {ci:<17} "
              f"{s['n_invalid']:>8} {pct(s['accuracy_on_valid']):>11}")
        if s["n_gold_unparsed"]:
            print(f"  [warn] {s['n_gold_unparsed']} gold answers could not be parsed in {name} — check the log format")

    print("\nAnswer-letter distribution (position-bias check):")
    for name, s in overall.items():
        print(f"  {name:<10} predicted: {s['pred_distribution']}")
    print(f"  {'gold':<10} actual:    {overall['llm_only']['gold_distribution']}")

    # ---- 2. Per dataset --------------------------------------------------- #
    by_ds_rows = []
    datasets = sorted({s["dataset"] for rows in systems.values() for s in rows})
    print("\nAccuracy by dataset:")
    print(f"  {'dataset':<10} {'n':>4} {'LLM-only':>9} {'RAG':>9} {'Δ (pts)':>9}")
    for d in datasets:
        rec = {"dataset": d}
        for name, rows in systems.items():
            sub = [s for s in rows if s["dataset"] == d]
            sm = summarize(sub) if sub else None
            rec[f"{name}_n"] = sm["n"] if sm else 0
            rec[f"{name}_accuracy"] = round(sm["accuracy"], 4) if sm else None
            rec[f"{name}_ci_low"] = round(sm["ci_low"], 4) if sm else None
            rec[f"{name}_ci_high"] = round(sm["ci_high"], 4) if sm else None
            rec[f"{name}_invalid"] = sm["n_invalid"] if sm else None
        a, b = rec["llm_only_accuracy"], rec["rag_accuracy"]
        rec["delta"] = round(b - a, 4) if a is not None and b is not None else None
        by_ds_rows.append(rec)
        delta = f"{100 * rec['delta']:+.1f}" if rec["delta"] is not None else "n/a"
        print(f"  {d:<10} {rec['llm_only_n']:>4} {pct(a) if a is not None else 'n/a':>9} "
              f"{pct(b) if b is not None else 'n/a':>9} {delta:>9}")
    if datasets == ["unknown"]:
        print("  (no dataset field found — rerun with --infer-dataset-by-position for the 50-question pilot)")

    # ---- 3. Paired comparison -------------------------------------------- #
    llm_by_key = {s["key"]: s for s in systems["llm_only"]}
    rag_by_key = {s["key"]: s for s in systems["rag"]}
    shared = [k for k in llm_by_key if k in rag_by_key]
    only_llm = len(llm_by_key) - len(shared)
    only_rag = len(rag_by_key) - len(shared)

    per_q, pairs = [], []
    counts = Counter()
    for k in shared:
        l, r = llm_by_key[k], rag_by_key[k]
        if l["correct"] and r["correct"]:
            cat = "both_correct"
        elif not l["correct"] and r["correct"]:
            cat = "rag_helped"
        elif l["correct"] and not r["correct"]:
            cat = "rag_hurt"
        else:
            cat = "both_wrong"
        counts[cat] += 1
        pairs.append((int(l["correct"]), int(r["correct"])))
        per_q.append({
            "key": k, "dataset": l["dataset"], "outcome": cat, "gold": l["gold"],
            "llm_pred": l["pred"], "rag_pred": r["pred"],
            "llm_raw": l["raw_answer"], "rag_raw": r["raw_answer"],
            "llm_confidence": l["confidence"], "rag_confidence": r["confidence"],
            "question": l["question"],
        })

    n_pair = len(shared)
    b, c = counts["rag_helped"], counts["rag_hurt"]
    p_val = mcnemar_exact(b, c)
    diff = (sum(r for _, r in pairs) - sum(l for l, _ in pairs)) / n_pair if n_pair else float("nan")
    d_lo, d_hi = bootstrap_diff_ci(pairs, args.bootstrap, args.seed)

    print(f"\nPaired comparison on {n_pair} shared questions"
          + (f"  ({only_llm} only in LLM log, {only_rag} only in RAG log)" if only_llm or only_rag else ""))
    print(f"  both correct : {counts['both_correct']}")
    print(f"  RAG helped   : {b}   (LLM wrong → RAG right)")
    print(f"  RAG hurt     : {c}   (LLM right → RAG wrong)")
    print(f"  both wrong   : {counts['both_wrong']}")
    print(f"  Accuracy difference (RAG − LLM): {100 * diff:+.1f} pts, "
          f"95% bootstrap CI [{100 * d_lo:+.1f}, {100 * d_hi:+.1f}]")
    print(f"  Exact McNemar p = {p_val:.4f}"
          + ("   (significant at α = 0.05)" if p_val < 0.05 else "   (not significant at α = 0.05)"))
    if n_pair < 100:
        print("  Note: with fewer than ~100 paired questions, only large differences can reach significance.")

    # ---- 4. Save ---------------------------------------------------------- #
    summary = {
        "overall": overall,
        "paired": {
            "n_shared": n_pair, "only_in_llm_log": only_llm, "only_in_rag_log": only_rag,
            **{k: counts[k] for k in ("both_correct", "rag_helped", "rag_hurt", "both_wrong")},
            "accuracy_difference": diff, "diff_ci_low": d_lo, "diff_ci_high": d_hi,
            "mcnemar_exact_p": p_val, "bootstrap_iters": args.bootstrap, "seed": args.seed,
        },
        "by_dataset": by_ds_rows,
        "inputs": {"llm": os.path.abspath(args.llm), "rag": os.path.abspath(args.rag)},
    }
    with open(os.path.join(args.out, "accuracy_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    write_csv(os.path.join(args.out, "accuracy_overall.csv"), [
        {"system": n, **{k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in s.items() if not isinstance(v, dict)}}
        for n, s in overall.items()
    ])
    write_csv(os.path.join(args.out, "accuracy_by_dataset.csv"), by_ds_rows)
    write_csv(os.path.join(args.out, "paired_outcomes.csv"), per_q)
    invalid = [s for rows in systems.values() for s in rows if not s["valid"]]
    write_csv(os.path.join(args.out, "invalid_answers.csv"),
              [{k: s[k] for k in ("system", "dataset", "key", "line", "raw_answer", "question")} for s in invalid])

    print(f"\nSaved to {args.out}/: accuracy_summary.json, accuracy_overall.csv, "
          f"accuracy_by_dataset.csv, paired_outcomes.csv"
          + (", invalid_answers.csv" if invalid else ""))
    return summary


if __name__ == "__main__":
    main()
