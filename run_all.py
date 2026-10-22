#!/usr/bin/env python3
"""
Run every MedTrustEval evaluation and write the paper's result tables.

    python run_all.py --llm logs/llm_only_results.jsonl --rag logs/rag_results.jsonl --out results

Optional:
    --infer-dataset-by-position       (50-question pilot logs without a "dataset" field)
    --judge lexical                   (pipeline test only, no torch/transformers needed)
    --llm-runs / --rag-runs / --llm-shuffled / --rag-shuffled   (consistency test; see evaluate_consistency.py)
    --skip-hallucination              (e.g. before torch/transformers are installed)

Writes results/RESULTS_TABLES.md (Tables 3–5 of the paper) plus each script's own files.
"""
from __future__ import annotations

import argparse
import datetime as dt

import evaluate_accuracy
import evaluate_calibration
import evaluate_consistency
import evaluate_hallucination_grounding
from mte_common import pct


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def p(x):
    return "—" if x is None else pct(x)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="logs/llm_only_results.jsonl")
    ap.add_argument("--rag", default="logs/rag_results.jsonl")
    ap.add_argument("--out", default="results")
    ap.add_argument("--infer-dataset-by-position", action="store_true")
    ap.add_argument("--judge", default="nli", choices=["nli", "lexical"])
    ap.add_argument("--skip-hallucination", action="store_true")
    ap.add_argument("--llm-runs")
    ap.add_argument("--rag-runs")
    ap.add_argument("--llm-shuffled")
    ap.add_argument("--rag-shuffled")
    args = ap.parse_args(argv)

    acc_args = ["--llm", args.llm, "--rag", args.rag, "--out", f"{args.out}/accuracy"]
    if args.infer_dataset_by_position:
        acc_args.append("--infer-dataset-by-position")
    acc = evaluate_accuracy.main(acc_args)

    hal = None
    if not args.skip_hallucination:
        hal = evaluate_hallucination_grounding.main(
            ["--llm", args.llm, "--rag", args.rag, "--out", f"{args.out}/hallucination", "--judge", args.judge])

    cal = evaluate_calibration.main(["--llm", args.llm, "--rag", args.rag, "--out", f"{args.out}/calibration"])

    con_args = ["--llm", args.llm, "--rag", args.rag, "--out", f"{args.out}/consistency"]
    for flag, val in (("--llm-runs", args.llm_runs), ("--rag-runs", args.rag_runs),
                      ("--llm-shuffled", args.llm_shuffled), ("--rag-shuffled", args.rag_shuffled)):
        if val:
            con_args += [flag, val]
    con = evaluate_consistency.main(con_args)

    # ------------------------------------------------------------------ tables
    o, pr = acc["overall"], acc["paired"]
    L = [f"# MedTrustEval results", "",
         f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} from `{args.llm}` and `{args.rag}`.", ""]
    if args.judge == "lexical" and hal:
        L += ["> **WARNING: hallucination/grounding numbers below come from the lexical TEST judge and must not be reported.**", ""]
    L += ["## Table 3. Accuracy", "",
          "| System | n | Accuracy (95% CI) | Invalid rate | Accuracy on valid answers |",
          "|---|---|---|---|---|"]
    for name, lab in (("llm_only", "LLM-only"), ("rag", "LLM + RAG")):
        s = o[name]
        L.append(f"| {lab} | {s['n']} | {p(s['accuracy'])} ({p(s['ci_low'])}–{p(s['ci_high'])}) | "
                 f"{p(s['invalid_rate'])} | {p(s['accuracy_on_valid'])} |")
    L += [f"| Difference (RAG − LLM) | {pr['n_shared']} paired | {100 * pr['accuracy_difference']:+.1f} pts "
          f"(95% CI {100 * pr['diff_ci_low']:+.1f} to {100 * pr['diff_ci_high']:+.1f}) | McNemar p = {pr['mcnemar_exact_p']:.3f} | |",
          "", f"Paired outcomes: both correct {pr['both_correct']}, RAG helped {pr['rag_helped']}, "
          f"RAG hurt {pr['rag_hurt']}, both wrong {pr['both_wrong']}.", "",
          f"Predicted letters — LLM-only: {o['llm_only']['pred_distribution']}; RAG: {o['rag']['pred_distribution']}; "
          f"correct answers: {o['llm_only']['gold_distribution']}.", "",
          "## Table 4. Accuracy by dataset", "",
          "| Dataset | n | LLM-only | LLM + RAG | Difference |", "|---|---|---|---|---|"]
    for d in acc["by_dataset"]:
        diff = "—" if d["delta"] is None else f"{100 * d['delta']:+.1f} pts"
        L.append(f"| {d['dataset']} | {d['llm_only_n']} | {p(d['llm_only_accuracy'])} | {p(d['rag_accuracy'])} | {diff} |")

    h = hal or {}
    hl, hr = h.get("llm_only", {}), h.get("rag", {})
    cl, cr = cal["llm_only"], cal["rag"]
    rl = con.get("llm_only", {}).get("repeat", {})
    rr = con.get("rag", {}).get("repeat", {})
    sl = con.get("llm_only", {}).get("shuffle", {})
    sr = con.get("rag", {}).get("shuffle", {})
    L += ["", "## Table 5. MedTrustEval profile", "",
          "| Dimension | Metric | LLM-only | LLM + RAG |", "|---|---|---|---|",
          f"| Accuracy | Accuracy ↑ | {p(o['llm_only']['accuracy'])} | {p(o['rag']['accuracy'])} |",
          f"| Hallucination | Contradicted-statement rate ↓ | {p(hl.get('contradicted_statement_rate'))} | {p(hr.get('contradicted_statement_rate'))} |",
          f"| Hallucination | Responses with ≥1 contradiction ↓ | {p(hl.get('response_hallucination_rate'))} | {p(hr.get('response_hallucination_rate'))} |",
          f"| Hallucination | Confident-error rate ↓ | {p(cl.get('confident_error_rate'))} | {p(cr.get('confident_error_rate'))} |",
          f"| Grounding | Faithfulness (supported statements) ↑ | {p(hl.get('supported_statement_rate'))}* | {p(hr.get('faithfulness'))} |",
          f"| Grounding | Fully grounded responses ↑ | {p(hl.get('fully_grounded_rate'))}* | {p(hr.get('fully_grounded_rate'))} |",
          f"| Retrieval | Evidence hit rate ↑ | n/a | {p(hr.get('evidence_hit_rate'))} |",
          f"| Retrieval | RAG accuracy with / without evidence hit | n/a | {p(hr.get('rag_accuracy_when_hit'))} / {p(hr.get('rag_accuracy_when_no_hit'))} |",
          f"| Retrieval | Precision@5 (manual) ↑ | n/a | see review/manual_review_scores.json |",
          f"| Citation | Validity ↑ | n/a | {p(hr.get('citation_validity'))} |",
          f"| Citation | Recall ↑ | n/a | {p(hr.get('citation_recall'))} |",
          f"| Citation | Precision ↑ | n/a | {p(hr.get('citation_precision'))} |",
          f"| Consistency | Answer agreement across runs ↑ | {p(rl.get('answer_agreement'))} | {p(rr.get('answer_agreement'))} |",
          f"| Consistency | Shuffle stability ↑ | {p(sl.get('shuffle_stability'))} | {p(sr.get('shuffle_stability'))} |",
          f"| Calibration | ECE ↓ | {f3(cl.get('ece'))} | {f3(cr.get('ece'))} |",
          f"| Calibration | Brier score ↓ | {f3(cl.get('brier'))} | {f3(cr.get('brier'))} |",
          f"| Calibration | Confidence gap (correct − wrong) ↑ | {p(cl.get('confidence_gap'))} | {p(cr.get('confidence_gap'))} |",
          "",
          "\\* For LLM-only answers, support is measured against the passages retrieved for the RAG system, "
          "which the LLM-only model never saw.",
          "",
          "— = not computed yet (missing input, e.g. repeated runs for consistency)."]
    path = f"{args.out}/RESULTS_TABLES.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n>>> Result tables written to {path}")


if __name__ == "__main__":
    main()
