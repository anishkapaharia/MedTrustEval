#!/usr/bin/env python3
"""
MedTrustEval evaluation — Dimensions 2–5
========================================
  2. Hallucination     (both systems)
  3. Evidence grounding (RAG)
  4. Retrieval quality  – automatic evidence-hit rate (RAG)
  5. Citation fidelity  (RAG)

How it works
------------
Each model explanation ("reasoning") is split into sentence-level statements.
Every statement is checked against the 5 passages MedRAG retrieved for that
question, using a natural-language-inference (NLI) model:

  supported      some passage entails the statement      (entailment >= --entail-threshold)
  contradicted   not supported, and some passage contradicts it (contradiction >= --contra-threshold)
  not addressed  neither (the textbook passages don't cover it) — NOT counted as hallucination

The SAME retrieved passages are used as reference evidence for the LLM-only
answers (the LLM-only model never saw them), so both systems are judged
against identical evidence.

Metrics
-------
Hallucination   contradicted-statement rate; response hallucination rate (>=1 contradicted);
                confident-error rate (wrong or invalid answer with confidence >= 80)
Grounding       faithfulness (% statements supported); fully-grounded responses (%)
Retrieval       evidence-hit rate (>=1 passage entails "the answer is <correct option>");
                RAG accuracy with vs. without an evidence hit
Citations       citation validity (every sentence cited, all markers in 1..k);
                citation recall (ALCE-style: % sentences supported by their cited passages);
                citation precision (% individual citations that are relevant, ALCE definition)

Usage
-----
    pip install torch transformers          # once
    python evaluate_hallucination_grounding.py \
        --llm logs/llm_only_results.jsonl --rag logs/rag_results.jsonl \
        --out results/hallucination

    # quick pipeline test without installing anything (numbers are meaningless):
    python evaluate_hallucination_grounding.py ... --judge lexical
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

from mte_common import (get_confidence, get_docs, get_key, get_reasoning, is_correct, load_jsonl,
                        normalize_answer, options_as_dict, pct, sentence_citations, split_sentences,
                        strip_citations, write_csv, write_json, get_dataset)
from nli_judge import DEFAULT_MODEL, make_judge

CONFIDENT = 0.80


def label_statement(scores: list[dict], te: float, tc: float):
    ent = max((s["entailment"] for s in scores), default=0.0)
    con = max((s["contradiction"] for s in scores), default=0.0)
    if ent >= te:
        lab = "supported"
    elif con >= tc:
        lab = "contradicted"
    else:
        lab = "not_addressed"
    best = max(range(len(scores)), key=lambda i: scores[i]["entailment"]) + 1 if scores else None
    return lab, ent, con, best


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", t.lower())).strip()


def is_verbatim_copy(statement: str, docs: list[str], min_chars: int = 40) -> bool:
    """True if the statement (ignoring case/punctuation) appears word-for-word in a passage.
    Copied sentences are trivially 'supported', so faithfulness is also reported without them."""
    s = _norm(statement)
    if len(s) < min_chars:
        return False
    return any(s in _norm(d) for d in docs)


def answer_hypothesis(row: dict) -> str | None:
    opts = options_as_dict(row.get("options"))
    gold = normalize_answer(row.get("correct_answer", row.get("answer")), row.get("options"))
    if not gold or gold not in opts:
        return None
    q = str(row.get("question", "")).strip()
    ans = opts[gold].strip()
    if ans.lower() in ("yes", "no", "maybe"):
        return f"Question: {q} The answer is {ans.lower()}."
    return f"Question: {q} The answer is: {ans}."


def evaluate(llm_rows, rag_rows, judge, te=0.5, tc=0.5, limit=None):
    rag_by_key = {get_key(r): r for r in rag_rows}
    systems = {"llm_only": llm_rows, "rag": rag_rows}
    stmt_rows, resp_rows, cite_rows, hit_rows = [], [], [], []
    missing_evidence = defaultdict(int)

    for system, rows in systems.items():
        rows = rows[:limit] if limit else rows
        # ---- collect all NLI pairs first (batched) ----
        jobs = []
        for r in rows:
            key = get_key(r)
            ev_row = rag_by_key.get(key)
            docs = get_docs(ev_row) if ev_row else []
            if not docs:
                missing_evidence[system] += 1
                continue
            sents = split_sentences(get_reasoning(r))
            jobs.append((r, key, docs, sents))
        pairs = []
        for r, key, docs, sents in jobs:
            for s in sents:
                h = strip_citations(s)
                pairs += [(d, h) for d in docs]
                if system == "rag":
                    cited = [c for c in sentence_citations(s) if 1 <= c <= len(docs)]
                    if len(cited) > 1:
                        pairs.append((" ".join(docs[c - 1] for c in cited), h))
                        for c in cited:
                            rest = [x for x in cited if x != c]
                            pairs.append((" ".join(docs[x - 1] for x in rest), h))
            if system == "rag":
                hyp = answer_hypothesis(r)
                if hyp:
                    pairs += [(d, hyp) for d in docs]
        scores = dict(zip(pairs, judge.score(pairs))) if pairs else {}

        def ent(p, h):
            return scores[(p, h)]["entailment"] >= te

        # ---- score ----
        for r, key, docs, sents in jobs:
            valid, correct = is_correct(r)
            conf = get_confidence(r)
            labels = []
            n_cited_ok = n_sent_supported_by_cites = 0
            all_markers_valid = bool(sents)
            for i, s in enumerate(sents, 1):
                h = strip_citations(s)
                lab, e, c, best = label_statement([scores[(d, h)] for d in docs], te, tc)
                labels.append(lab)
                row = {"system": system, "key": key, "dataset": get_dataset(r) or "", "sent_idx": i,
                       "statement": h, "label": lab, "max_entail": round(e, 3), "max_contra": round(c, 3),
                       "best_doc": best, "verbatim_copy": is_verbatim_copy(h, docs)}
                if system == "rag":
                    marks = sentence_citations(s)
                    cited = [m for m in marks if 1 <= m <= len(docs)]
                    bad = [m for m in marks if not 1 <= m <= len(docs)]
                    if not marks or bad:
                        all_markers_valid = False
                    supported_by_cites = False
                    if cited:
                        n_cited_ok += 1
                        joint = " ".join(docs[x - 1] for x in cited)
                        supported_by_cites = any(ent(docs[x - 1], h) for x in cited) or \
                            (len(cited) > 1 and ent(joint, h))
                        for m in cited:
                            alone = ent(docs[m - 1], h)
                            if alone:
                                relevant = True
                            elif len(cited) > 1 and supported_by_cites:
                                rest = " ".join(docs[x - 1] for x in cited if x != m)
                                relevant = not ent(rest, h)   # removing it breaks support
                            else:
                                relevant = False
                            cite_rows.append({"key": key, "sent_idx": i, "cited_doc": m, "relevant": relevant})
                    n_sent_supported_by_cites += supported_by_cites
                    row.update({"citations": marks, "invalid_markers": bad,
                                "supported_by_citations": supported_by_cites})
                stmt_rows.append(row)

            n = len(labels)
            resp = {
                "system": system, "key": key, "dataset": get_dataset(r) or "",
                "n_statements": n,
                "n_supported": labels.count("supported"),
                "n_contradicted": labels.count("contradicted"),
                "n_not_addressed": labels.count("not_addressed"),
                "hallucinated": labels.count("contradicted") > 0,
                "fully_grounded": n > 0 and labels.count("supported") == n,
                "valid": valid, "correct": correct, "confidence": conf,
                "confident_error": bool(not correct and conf is not None and conf >= CONFIDENT),
            }
            if system == "rag":
                resp.update({
                    "citation_valid": all_markers_valid,
                    "logged_citations_valid": r.get("citations_valid"),
                    "citation_recall": (n_sent_supported_by_cites / n) if n else None,
                })
                hyp = answer_hypothesis(r)
                if hyp:
                    hit_docs = [j + 1 for j, d in enumerate(docs) if ent(d, hyp)]
                    hit_rows.append({"key": key, "dataset": get_dataset(r) or "", "evidence_hit": bool(hit_docs),
                                     "hit_docs": hit_docs, "correct": correct})
            resp_rows.append(resp)

    return stmt_rows, resp_rows, cite_rows, hit_rows, dict(missing_evidence)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(stmt_rows, resp_rows, cite_rows, hit_rows):
    out = {}
    for system in ("llm_only", "rag"):
        S = [s for s in stmt_rows if s["system"] == system]
        R = [r for r in resp_rows if r["system"] == system]
        if not R:
            continue
        m = {
            "n_responses": len(R),
            "n_statements": len(S),
            "empty_explanations": sum(r["n_statements"] == 0 for r in R),
            "supported_statement_rate": _mean([s["label"] == "supported" for s in S]),
            "contradicted_statement_rate": _mean([s["label"] == "contradicted" for s in S]),
            "not_addressed_statement_rate": _mean([s["label"] == "not_addressed" for s in S]),
            "response_hallucination_rate": _mean([r["hallucinated"] for r in R]),
            "fully_grounded_rate": _mean([r["fully_grounded"] for r in R]),
            "confident_error_rate": _mean([r["confident_error"] for r in R]),
        }
        if system == "rag":
            m["faithfulness"] = m["supported_statement_rate"]
            nc = [x for x in S if not x.get("verbatim_copy")]
            m["verbatim_copy_rate"] = _mean([bool(x.get("verbatim_copy")) for x in S])
            m["faithfulness_excluding_copies"] = _mean([x["label"] == "supported" for x in nc])
            m["citation_validity"] = _mean([r["citation_valid"] for r in R])
            logged = [r["logged_citations_valid"] for r in R if isinstance(r["logged_citations_valid"], bool)]
            m["logged_citation_validity"] = _mean(logged) if logged else None
            m["citation_recall"] = _mean([s["supported_by_citations"] for s in S])
            m["citation_precision"] = _mean([c["relevant"] for c in cite_rows])
            m["n_citations"] = len(cite_rows)
            if hit_rows:
                hit = [h for h in hit_rows if h["evidence_hit"]]
                miss = [h for h in hit_rows if not h["evidence_hit"]]
                m["evidence_hit_rate"] = len(hit) / len(hit_rows)
                m["rag_accuracy_when_hit"] = _mean([h["correct"] for h in hit])
                m["rag_accuracy_when_no_hit"] = _mean([h["correct"] for h in miss])
                m["n_hit"], m["n_no_hit"] = len(hit), len(miss)
        out[system] = m
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="logs/llm_only_results.jsonl")
    ap.add_argument("--rag", default="logs/rag_results.jsonl")
    ap.add_argument("--out", default="results/hallucination")
    ap.add_argument("--judge", choices=["nli", "lexical"], default="nli")
    ap.add_argument("--nli-model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None, help="cpu / cuda / mps (auto if omitted)")
    ap.add_argument("--entail-threshold", type=float, default=0.5)
    ap.add_argument("--contra-threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions per system")
    args = ap.parse_args(argv)

    judge = make_judge(args.judge, args.nli_model, args.device)
    if args.judge == "lexical":
        print("  [warn] lexical judge = pipeline test only. Do NOT report these numbers.", file=sys.stderr)

    llm, rag = load_jsonl(args.llm), load_jsonl(args.rag)
    stmts, resps, cites, hits, missing = evaluate(llm, rag, judge, args.entail_threshold,
                                                  args.contra_threshold, args.limit)
    summary = summarize(stmts, resps, cites, hits)
    summary["settings"] = {"judge": args.judge, "nli_model": getattr(judge, "model_name", None),
                           "entail_threshold": args.entail_threshold,
                           "contra_threshold": args.contra_threshold,
                           "confident_threshold": CONFIDENT, "questions_without_evidence": missing}

    print("\n=== MedTrustEval · Hallucination, Grounding, Retrieval, Citations ===")
    print(f"judge: {summary['settings']['nli_model']}\n")
    rows = [
        ("Statements checked", "n_statements", False),
        ("Supported statements (faithfulness)", "supported_statement_rate", True),
        ("Sentences copied word-for-word from a passage", "verbatim_copy_rate", True),
        ("Faithfulness excluding copied sentences", "faithfulness_excluding_copies", True),
        ("Contradicted statements", "contradicted_statement_rate", True),
        ("Not addressed by evidence", "not_addressed_statement_rate", True),
        ("Responses with >=1 contradiction", "response_hallucination_rate", True),
        ("Fully grounded responses", "fully_grounded_rate", True),
        ("Confident errors (wrong, conf >= 80)", "confident_error_rate", True),
        ("Citation validity", "citation_validity", True),
        ("Citation validity (logged flag)", "logged_citation_validity", True),
        ("Citation recall", "citation_recall", True),
        ("Citation precision", "citation_precision", True),
        ("Evidence hit rate", "evidence_hit_rate", True),
        ("RAG accuracy | evidence hit", "rag_accuracy_when_hit", True),
        ("RAG accuracy | no evidence hit", "rag_accuracy_when_no_hit", True),
    ]
    print(f"{'metric':<40} {'LLM-only':>10} {'RAG':>10}")
    for label, k, is_pct in rows:
        vals = []
        for sysname in ("llm_only", "rag"):
            v = summary.get(sysname, {}).get(k)
            vals.append("—" if v is None else (pct(v) if is_pct else str(v)))
        print(f"{label:<40} {vals[0]:>10} {vals[1]:>10}")
    if missing:
        print(f"\n  [warn] questions with no retrieved documents to check against: {missing}")

    write_json(f"{args.out}/hallucination_summary.json", summary)
    write_csv(f"{args.out}/statements.csv", stmts)
    write_csv(f"{args.out}/responses.csv", resps)
    write_csv(f"{args.out}/citations.csv", cites)
    write_csv(f"{args.out}/evidence_hits.csv", hits)
    print(f"\nSaved to {args.out}/: hallucination_summary.json, statements.csv, responses.csv, "
          f"citations.csv, evidence_hits.csv")
    return summary


if __name__ == "__main__":
    main()
