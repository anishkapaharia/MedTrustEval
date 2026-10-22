#!/usr/bin/env python3
"""
MedTrustEval evaluation — Dimension 7: Calibration
==================================================
Does the model's self-reported confidence (0–100) match how often it is right?

Metrics (per system)
  ECE            Expected Calibration Error, 10 equal-width bins (Guo et al., 2017). Lower is better.
  Brier score    mean (confidence − correct)^2 (Brier, 1950). Lower is better.
  Mean confidence vs. accuracy  (overconfidence = mean confidence − accuracy)
  Confidence gap mean confidence on correct answers − on wrong answers. Higher is better.
  Confident-error rate  wrong answers given with confidence >= 80.

Invalid answers are counted as wrong. Rows with no usable confidence are skipped
and counted. A reliability diagram is saved if matplotlib is installed.

Usage
    python evaluate_calibration.py --llm logs/llm_only_results.jsonl --rag logs/rag_results.jsonl \
        --out results/calibration
"""
from __future__ import annotations

import argparse

from mte_common import get_confidence, is_correct, load_jsonl, pct, write_csv, write_json

N_BINS = 10


def calibration(pairs: list[tuple[float, bool]], n_bins: int = N_BINS) -> dict:
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    bins = []
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        inb = [(c, y) for c, y in pairs if (lo <= c < hi) or (b == n_bins - 1 and c == 1.0)]
        if inb:
            conf = sum(c for c, _ in inb) / len(inb)
            acc = sum(y for _, y in inb) / len(inb)
            ece += len(inb) / n * abs(acc - conf)
        else:
            conf = acc = None
        bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(inb), "mean_conf": conf, "accuracy": acc})
    right = [c for c, y in pairs if y]
    wrong = [c for c, y in pairs if not y]
    mean_conf = sum(c for c, _ in pairs) / n
    acc = sum(y for _, y in pairs) / n
    return {
        "n": n,
        "accuracy": acc,
        "mean_confidence": mean_conf,
        "overconfidence": mean_conf - acc,
        "ece": ece,
        "brier": sum((c - y) ** 2 for c, y in pairs) / n,
        "mean_conf_correct": sum(right) / len(right) if right else None,
        "mean_conf_wrong": sum(wrong) / len(wrong) if wrong else None,
        "confidence_gap": (sum(right) / len(right) - sum(wrong) / len(wrong)) if right and wrong else None,
        "confident_error_rate": sum(1 for c, y in pairs if not y and c >= 0.8) / n,
        "distinct_confidence_values": len({round(c, 2) for c, _ in pairs}),
        "bins": bins,
    }


def plot(results: dict, path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], ls="--", c="#999", lw=1, label="Perfect calibration")
    colors = {"llm_only": "#4C72B0", "rag": "#DD8452"}
    names = {"llm_only": "LLM-only", "rag": "LLM + RAG"}
    for sysname, r in results.items():
        pts = [(b["mean_conf"], b["accuracy"]) for b in r.get("bins", []) if b["n"]]
        if pts:
            ax.plot(*zip(*pts), marker="o", c=colors.get(sysname), label=f"{names.get(sysname, sysname)} (ECE {r['ece']:.2f})")
    ax.set_xlabel("Self-reported confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="MedTrustEval calibration")
    ap.add_argument("--llm", default="logs/llm_only_results.jsonl")
    ap.add_argument("--rag", default="logs/rag_results.jsonl")
    ap.add_argument("--out", default="results/calibration")
    ap.add_argument("--bins", type=int, default=N_BINS)
    args = ap.parse_args(argv)

    results, skipped = {}, {}
    for sysname, path in (("llm_only", args.llm), ("rag", args.rag)):
        pairs, miss = [], 0
        for r in load_jsonl(path):
            c = get_confidence(r)
            if c is None:
                miss += 1
                continue
            _, correct = is_correct(r)
            pairs.append((c, correct))
        results[sysname] = calibration(pairs, args.bins)
        skipped[sysname] = miss

    print("\n=== MedTrustEval · Calibration ===\n")
    print(f"{'metric':<34} {'LLM-only':>10} {'RAG':>10}")
    for label, k, is_pct in [("n (with confidence)", "n", False), ("Accuracy", "accuracy", True),
                             ("Mean confidence", "mean_confidence", True),
                             ("Overconfidence (conf − acc)", "overconfidence", True),
                             ("ECE ↓", "ece", False), ("Brier ↓", "brier", False),
                             ("Confidence | correct", "mean_conf_correct", True),
                             ("Confidence | wrong", "mean_conf_wrong", True),
                             ("Confidence gap ↑", "confidence_gap", True),
                             ("Confident-error rate ↓", "confident_error_rate", True),
                             ("Distinct confidence values", "distinct_confidence_values", False)]:
        vals = []
        for s in ("llm_only", "rag"):
            v = results[s].get(k)
            vals.append("—" if v is None else (pct(v) if is_pct else (f"{v:.3f}" if isinstance(v, float) else str(v))))
        print(f"{label:<34} {vals[0]:>10} {vals[1]:>10}")
    if any(skipped.values()):
        print(f"\n  [warn] rows without a usable confidence (skipped): {skipped}")
    for s, r in results.items():
        if r.get("distinct_confidence_values", 99) <= 3:
            print(f"  [note] {s} uses only {r['distinct_confidence_values']} distinct confidence values — "
                  "calibration metrics will be coarse.")

    write_json(f"{args.out}/calibration_summary.json", {"results": results, "skipped_no_confidence": skipped})
    write_csv(f"{args.out}/calibration_bins.csv",
              [{"system": s, **b} for s, r in results.items() for b in r.get("bins", [])])
    made = plot(results, f"{args.out}/reliability_diagram.png")
    print(f"\nSaved to {args.out}/: calibration_summary.json, calibration_bins.csv"
          + (", reliability_diagram.png" if made else "  (install matplotlib for the diagram)"))
    return results


if __name__ == "__main__":
    main()
