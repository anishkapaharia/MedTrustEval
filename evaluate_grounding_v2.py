#!/usr/bin/env python3
"""
MedTrustEval — Dimensions 2-5, version 2 (fixed NLI judging)

Why this file exists
--------------------
Version 1 (`evaluate_hallucination_grounding.py`) produced results that were measuring the
judge rather than the systems: faithfulness 2.2% / 8.4% and an evidence-hit rate of exactly
0.0%. Diagnosis of the raw scores showed three separate faults, and only one of them was a
threshold problem (it wasn't, really - see below).

FAULT 1 - premise too long.
  The judge was asked "does this 136-word textbook paragraph entail this 20-word claim?".
  NLI models (SNLI/MNLI-trained) expect a SINGLE SENTENCE as the premise. On multi-sentence
  premises the entailment head under-fires: scores came back saturated, 96% of statements
  below 0.1 and a handful at 0.99, with almost nothing in between. Lowering the threshold
  from 0.5 to 0.1 moved faithfulness only from 8.4% to 12.3%, which is how we know the cut
  point was never the problem.
  FIX: split each passage into sentences and score the claim against sliding windows of
  1 and 2 sentences, taking the best window. This is standard practice (ALCE, RAGAS, FActScore
  all score against short evidence spans, not whole documents).

FAULT 2 - the evidence-hit hypothesis was unanswerable.
  v1 asked whether a passage entails "Question: <200-word clinical vignette> The answer is:
  <option>". No NLI model entails a hypothesis of that shape, so the metric returned 0/50.
  FIX: the hypothesis is now the answer claim on its own. Also reports a lexical
  answer-presence check, which needs no model at all and is what most retrieval papers use.

FAULT 3 - two kinds of statement should never have been judged.
  (a) Negation. Statements containing "not/no/never/without" were labelled "contradicted"
      39-44% of the time against 17-18% for statements without - the well-known negation
      artifact of NLI models, not a real hallucination signal.
  (b) Meta-statements. Sentences like "the documents do not mention pembrolizumab" are
      claims ABOUT the evidence, not medical claims. 8 of 16 were scored "contradicted".
  FIX: both are flagged per statement. Meta-statements are excluded from the grounding
  metrics and counted separately; negation-bearing statements are kept but reported as a
  separate rate so the artifact is visible rather than buried.

Also changed: the default judge is now MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, which
adds FEVER (evidence-based fact verification) to the training mix and is a much closer match
to "is this claim supported by this passage?" than plain MNLI. Same size, same speed.

NONE OF THIS MAKES THE NUMBERS TRUE. An automatic judge has to be validated against human
labels before its output goes in a paper. Run `review_sheets.py` / the Day 9 workbook, label
a sample by hand, and report the agreement. If agreement is poor, that is a finding worth
reporting, not a problem to hide.

Usage
-----
    python3 evaluate_grounding_v2.py \
        --llm logs/v2_llama32_3b/repeats/llm_only_run1.jsonl \
        --rag logs/v2_llama32_3b/repeats/rag_run1.jsonl \
        --out results/grounding_v2 --device cpu

    # to reproduce the v1 behaviour for comparison:
    python3 evaluate_grounding_v2.py ... --window-sizes 0 --nli-model cross-encoder/nli-deberta-v3-base
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

from mte_common import (get_confidence, get_docs, get_key, get_reasoning, is_correct, load_jsonl,
                        normalize_answer, options_as_dict, pct, sentence_citations, split_sentences,
                        strip_citations, write_csv, write_json, get_dataset)
from nli_judge import make_judge

CONFIDENT = 0.80
DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

NEGATION = re.compile(
    r"\b(not|no|never|without|cannot|can't|isn't|aren't|doesn't|don't|didn't|"
    r"unlikely|rather than|instead of|rule out|absence of|neither|nor)\b", re.I)
META = re.compile(r"\b(document|passage|text|source|reference|context|excerpt)s?\b", re.I)


# ---------------------------------------------------------------- evidence spans
def windows(text: str, sizes=(1, 2), cap: int = 24) -> list[str]:
    """Sliding sentence windows over one passage. sizes=() means 'whole passage only'."""
    sents = split_sentences(text)
    if not sents:
        return [text] if text.strip() else []
    out = []
    for k in sizes:
        for i in range(len(sents) - k + 1):
            out.append(" ".join(sents[i:i + k]))
    if not sizes:
        out = [text]
    # always keep the whole passage as a fallback span
    if text not in out:
        out.append(text)
    return out[:cap] if len(out) > cap else out


def answer_claim(row: dict) -> str | None:
    """The claim we want the evidence to support. v1 prepended the whole question, which no
    NLI model will ever entail; here it is the answer on its own."""
    opts = options_as_dict(row.get("options"))
    gold = normalize_answer(row.get("correct_answer", row.get("answer")), row.get("options"))
    if not gold or gold not in opts:
        return None
    ans = opts[gold].strip()
    if ans.lower() in ("yes", "no", "maybe"):
        # A bare "yes" is not a checkable claim. Use the question as the claim when the
        # gold answer is yes; skip otherwise, and say so in the summary.
        q = str(row.get("question", "")).strip()
        return q if ans.lower() == "yes" and q else None
    return ans


_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("""a an the of and or to in on for with by is are was were be been it this that these those as at
from which who what when where why how not no than then there their its his her they them we you i can may
might will would should could has have had do does did most more less very also such into""".split())


def lexical_answer_presence(row: dict, docs: list[str], min_overlap: float = 0.6) -> bool:
    """Model-free retrieval check: do the passages contain the content words of the gold answer?
    This is the 'answer presence / lexical recall' measure used in most retrieval papers."""
    opts = options_as_dict(row.get("options"))
    gold = normalize_answer(row.get("correct_answer", row.get("answer")), row.get("options"))
    if not gold or gold not in opts:
        return False
    ans = opts[gold].strip().lower()
    if ans in ("yes", "no", "maybe"):
        return False
    terms = {w for w in _WORD.findall(ans) if w not in _STOP and len(w) > 2}
    if not terms:
        return False
    blob = " ".join(docs).lower()
    hits = sum(1 for t in terms if t in blob)
    return hits / len(terms) >= min_overlap


# ---------------------------------------------------------------- labelling
def label_from(scores: list[dict], te: float, tc: float):
    """FIXED in v2.1. The previous rule tested entailment first, so a statement where one
    passage entails it and ANOTHER passage contradicts it was always called 'supported',
    even when the contradiction signal was much the stronger of the two. On the 3B run that
    affected 13% of all statements, and 35% of everything labelled 'supported' had a higher
    contradiction score than entailment score. Those are now their own category.

    'conflicting' is not a nuisance bucket - it is a real retrieval-quality signal. It means
    the five passages handed to the model disagree with each other about this claim, which is
    something a trustworthy system ought to surface to its user rather than silently pick a side."""
    ent = max((s["entailment"] for s in scores), default=0.0)
    con = max((s["contradiction"] for s in scores), default=0.0)
    if ent >= te and con >= tc:
        lab = "conflicting"
    elif ent >= te:
        lab = "supported"
    elif con >= tc:
        lab = "contradicted"
    else:
        lab = "not_addressed"
    return lab, ent, con


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").lower()).strip()


def is_verbatim_copy(statement: str, docs: list[str], min_chars: int = 40) -> bool:
    s = _norm(statement)
    if len(s) < min_chars:
        return False
    return any(s in _norm(d) for d in docs)


# ---------------------------------------------------------------- main evaluation
def evaluate(llm_rows, rag_rows, judge, te=0.5, tc=0.5, limit=None, sizes=(1, 2)):
    rag_by_key = {get_key(r): r for r in rag_rows}
    systems = {"llm_only": llm_rows, "rag": rag_rows}
    stmt_rows, resp_rows, cite_rows, hit_rows = [], [], [], []
    missing_evidence = defaultdict(int)
    span_cache: dict[str, list[str]] = {}

    def spans(doc: str) -> list[str]:
        if doc not in span_cache:
            span_cache[doc] = windows(doc, sizes)
        return span_cache[doc]

    for system, rows in systems.items():
        rows = rows[:limit] if limit else rows
        jobs = []
        for r in rows:
            key = get_key(r)
            ev_row = rag_by_key.get(key)
            docs = get_docs(ev_row) if ev_row else []
            if not docs:
                missing_evidence[system] += 1
                continue
            jobs.append((r, key, docs, split_sentences(get_reasoning(r))))

        # ---- collect every (premise span, hypothesis) pair, then score in one batch ----
        pairs = []
        for r, key, docs, sents in jobs:
            for s in sents:
                h = strip_citations(s)
                if META.search(h):          # not a medical claim; excluded from grounding
                    continue
                for d in docs:
                    pairs += [(w, h) for w in spans(d)]
            if system == "rag":
                claim = answer_claim(r)
                if claim:
                    for d in docs:
                        pairs += [(w, claim) for w in spans(d)]
        pairs = list(dict.fromkeys(pairs))
        print(f"  {system}: {len(jobs)} responses, {len(pairs)} evidence-span comparisons "
              f"(window sizes {sizes or 'whole passage'})", file=sys.stderr)
        scores = dict(zip(pairs, judge.score(pairs))) if pairs else {}

        def best(doc: str, h: str) -> list[dict]:
            return [scores[(w, h)] for w in spans(doc) if (w, h) in scores]

        def supports(doc: str, h: str) -> bool:
            sc = best(doc, h)
            return bool(sc) and max(s["entailment"] for s in sc) >= te

        # ---- score ----
        for r, key, docs, sents in jobs:
            valid, correct = is_correct(r)
            conf = get_confidence(r)
            labels = []
            n_meta = 0
            n_sent_supported_by_cites = 0
            all_markers_valid = bool(sents)
            for i, s in enumerate(sents, 1):
                h = strip_citations(s)
                meta = bool(META.search(h))
                neg = bool(NEGATION.search(h))
                if meta:
                    n_meta += 1
                    lab, e, c = "meta_statement", 0.0, 0.0
                else:
                    flat = [sc for d in docs for sc in best(d, h)]
                    lab, e, c = label_from(flat, te, tc)
                    labels.append(lab)
                row = {"system": system, "key": key, "dataset": get_dataset(r) or "", "sent_idx": i,
                       "statement": h, "label": lab, "max_entail": round(e, 3), "max_contra": round(c, 3),
                       "meta_statement": meta, "has_negation": neg,
                       "verbatim_copy": is_verbatim_copy(h, docs)}
                if system == "rag":
                    marks = sentence_citations(s)
                    cited = [m for m in marks if 1 <= m <= len(docs)]
                    bad = [m for m in marks if not 1 <= m <= len(docs)]
                    if not marks or bad:
                        all_markers_valid = False
                    supported_by_cites = False
                    if cited and not meta:
                        supported_by_cites = any(supports(docs[x - 1], h) for x in cited)
                        for m in cited:
                            cite_rows.append({"key": key, "sent_idx": i, "cited_doc": m,
                                              "relevant": supports(docs[m - 1], h)})
                    if not meta:
                        n_sent_supported_by_cites += supported_by_cites
                    row.update({"citations": marks, "invalid_markers": bad,
                                "supported_by_citations": supported_by_cites})
                stmt_rows.append(row)

            n = len(labels)
            resp = {
                "system": system, "key": key, "dataset": get_dataset(r) or "",
                "n_statements": n, "n_meta_statements": n_meta,
                "n_supported": labels.count("supported"),
                "n_contradicted": labels.count("contradicted"),
                "n_not_addressed": labels.count("not_addressed"),
                "hallucinated": labels.count("contradicted") > 0,
                "fully_grounded": n > 0 and labels.count("supported") == n,
                "valid": valid, "correct": correct, "confidence": conf,
                "confident_error": bool(not correct and conf is not None and conf >= CONFIDENT),
            }
            if system == "rag":
                claim = answer_claim(r)
                hit_docs = [j + 1 for j, d in enumerate(docs) if claim and supports(d, claim)] if claim else []
                resp.update({"citation_valid": all_markers_valid,
                             "logged_citations_valid": r.get("citations_valid"),
                             "citation_recall": (n_sent_supported_by_cites / n) if n else None})
                hit_rows.append({"key": key, "dataset": get_dataset(r) or "",
                                 "claim_checkable": bool(claim),
                                 "evidence_hit": bool(hit_docs), "hit_docs": hit_docs,
                                 "lexical_answer_present": lexical_answer_presence(r, docs),
                                 "correct": correct})
            resp_rows.append(resp)

    return stmt_rows, resp_rows, cite_rows, hit_rows, dict(missing_evidence)


# ---------------------------------------------------------------- summary
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def score_histogram(stmts, field="max_entail"):
    """Makes the v1 failure visible: if the scores are saturated at 0 and 1 with nothing in
    between, the judge is not uncertain - it is out of its depth, and moving the threshold
    will not help."""
    bins = {"0.0-0.1": 0, "0.1-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0}
    for r in stmts:
        v = float(r[field])
        for name, (lo, hi) in zip(bins, [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]):
            if lo <= v < hi:
                bins[name] += 1
                break
    return bins


def summarize(stmt_rows, resp_rows, cite_rows, hit_rows):
    out = {}
    for system in ("llm_only", "rag"):
        stmts = [s for s in stmt_rows if s["system"] == system and not s["meta_statement"]]
        metas = [s for s in stmt_rows if s["system"] == system and s["meta_statement"]]
        resps = [r for r in resp_rows if r["system"] == system]
        if not resps:
            continue
        n = len(stmts)
        m = {
            "n_responses": len(resps),
            "n_statements_judged": n,
            "n_meta_statements_excluded": len(metas),
            "supported_statement_rate": sum(s["label"] == "supported" for s in stmts) / n if n else None,
            "contradicted_statement_rate": sum(s["label"] == "contradicted" for s in stmts) / n if n else None,
            "not_addressed_statement_rate": sum(s["label"] == "not_addressed" for s in stmts) / n if n else None,
            "conflicting_statement_rate": sum(s["label"] == "conflicting" for s in stmts) / n if n else None,
            "response_hallucination_rate": _mean([r["hallucinated"] for r in resps]),
            "fully_grounded_rate": _mean([r["fully_grounded"] for r in resps]),
            "confident_error_rate": _mean([r["confident_error"] for r in resps]),
            "entailment_histogram": score_histogram(stmts, "max_entail"),
            "contradiction_histogram": score_histogram(stmts, "max_contra"),
        }
        # negation artifact check - report it every time, so it can never hide again
        neg = [s for s in stmts if s["has_negation"]]
        pos = [s for s in stmts if not s["has_negation"]]
        m["negation_check"] = {
            "n_with_negation": len(neg),
            "contradicted_rate_with_negation": (sum(s["label"] == "contradicted" for s in neg) / len(neg)) if neg else None,
            "contradicted_rate_without_negation": (sum(s["label"] == "contradicted" for s in pos) / len(pos)) if pos else None,
        }
        if system == "rag":
            m["faithfulness"] = m["supported_statement_rate"]
            copies = [s for s in stmts if s["verbatim_copy"]]
            m["verbatim_copy_rate"] = len(copies) / n if n else None
            noncopy = [s for s in stmts if not s["verbatim_copy"]]
            m["faithfulness_excluding_copies"] = (
                sum(s["label"] == "supported" for s in noncopy) / len(noncopy)) if noncopy else None
            m["citation_validity"] = _mean([r["citation_valid"] for r in resps])
            m["logged_citation_validity"] = _mean([r["logged_citations_valid"] for r in resps])
            m["citation_recall"] = _mean([r["citation_recall"] for r in resps])
            m["citation_precision"] = _mean([c["relevant"] for c in cite_rows]) if cite_rows else None
            m["n_citations"] = len(cite_rows)
            checkable = [h for h in hit_rows if h["claim_checkable"]]
            m["n_questions_with_checkable_answer_claim"] = len(checkable)
            m["evidence_hit_rate"] = _mean([h["evidence_hit"] for h in checkable]) if checkable else None
            m["lexical_answer_presence_rate"] = _mean([h["lexical_answer_present"] for h in checkable]) if checkable else None
            for flag, name in [(True, "hit"), (False, "no_hit")]:
                g = [h for h in checkable if h["evidence_hit"] is flag]
                m[f"rag_accuracy_when_{name}"] = _mean([h["correct"] for h in g]) if g else None
                m[f"n_{name}"] = len(g)
        out[system] = m
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="logs/llm_only_results.jsonl")
    ap.add_argument("--rag", default="logs/rag_results.jsonl")
    ap.add_argument("--out", default="results/grounding_v2")
    ap.add_argument("--judge", choices=["nli", "lexical"], default="nli")
    ap.add_argument("--nli-model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None, help="cpu / mps / cuda. Use cpu on an 8 GB Mac.")
    ap.add_argument("--entail-threshold", type=float, default=0.5)
    ap.add_argument("--contra-threshold", type=float, default=0.5)
    ap.add_argument("--window-sizes", default="1,2",
                    help="sentence-window sizes for the evidence span; '0' = whole passage (v1 behaviour)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    sizes = tuple(int(x) for x in a.window_sizes.split(",") if x.strip() and int(x) > 0)
    llm, rag = load_jsonl(a.llm), load_jsonl(a.rag)
    judge = make_judge(a.judge, a.nli_model, a.device)
    print(f"judge: {getattr(judge, 'model_name', a.judge)}   windows: {sizes or 'whole passage'}", file=sys.stderr)

    stmts, resps, cites, hits, missing = evaluate(llm, rag, judge, a.entail_threshold,
                                                  a.contra_threshold, a.limit, sizes)
    summary = summarize(stmts, resps, cites, hits)
    summary["settings"] = {"judge": a.judge, "nli_model": getattr(judge, "model_name", a.judge),
                           "entail_threshold": a.entail_threshold, "contra_threshold": a.contra_threshold,
                           "window_sizes": list(sizes), "confident_threshold": CONFIDENT,
                           "questions_without_evidence": missing,
                           "version": "v2.1 (sentence-window premises, answer-claim hypothesis, "
                                      "meta-statements excluded, negation flagged, "
                                      "conflicting evidence no longer silently counted as supported)"}

    rows = [
        ("Statements judged", "n_statements_judged", False),
        ("Meta-statements excluded", "n_meta_statements_excluded", False),
        ("Supported statements (faithfulness)", "supported_statement_rate", True),
        ("Contradicted statements", "contradicted_statement_rate", True),
        ("Not addressed by evidence", "not_addressed_statement_rate", True),
        ("Passages CONFLICT about it", "conflicting_statement_rate", True),
        ("Responses with >=1 contradiction", "response_hallucination_rate", True),
        ("Fully grounded responses", "fully_grounded_rate", True),
        ("Confident errors (wrong, conf >= 80)", "confident_error_rate", True),
        ("Citation validity", "citation_validity", True),
        ("Citation recall", "citation_recall", True),
        ("Citation precision", "citation_precision", True),
        ("Evidence hit rate (NLI)", "evidence_hit_rate", True),
        ("Answer present in passages (lexical)", "lexical_answer_presence_rate", True),
        ("RAG accuracy | evidence hit", "rag_accuracy_when_hit", True),
        ("RAG accuracy | no evidence hit", "rag_accuracy_when_no_hit", True),
    ]
    print("\n=== MedTrustEval v2 · Hallucination, Grounding, Retrieval, Citations ===")
    print(f"judge: {summary['settings']['nli_model']}   windows: {summary['settings']['window_sizes'] or 'whole passage'}\n")
    print(f"{'metric':<42}{'LLM-only':>11}{'RAG':>11}")
    for label, kkey, is_pct in rows:
        a_ = summary.get("llm_only", {}).get(kkey)
        b_ = summary.get("rag", {}).get(kkey)
        f = (lambda v: "—" if v is None else (pct(v) if is_pct else str(v)))
        print(f"{label:<42}{f(a_):>11}{f(b_):>11}")

    print("\n--- judge health check (read this before trusting anything above) ---")
    for system in ("llm_only", "rag"):
        s = summary.get(system)
        if not s:
            continue
        h = s["entailment_histogram"]
        mid = h["0.1-0.3"] + h["0.3-0.5"] + h["0.5-0.7"]
        tot = sum(h.values()) or 1
        print(f"  {system}: entailment scores {h}")
        print(f"    {100*mid/tot:.0f}% of scores are in the middle of the range. If this is near 0,"
              f" the judge is saturated and the threshold is not the problem.")
        nc = s["negation_check"]
        if nc["contradicted_rate_with_negation"] is not None:
            print(f"    negation artifact: contradicted {pct(nc['contradicted_rate_with_negation'])} "
                  f"with a negation word vs {pct(nc['contradicted_rate_without_negation'])} without "
                  f"({nc['n_with_negation']} statements). A large gap means the contradiction rate is inflated.")

    write_json(f"{a.out}/grounding_summary.json", summary)
    write_csv(f"{a.out}/statements.csv", stmts)
    write_csv(f"{a.out}/responses.csv", resps)
    write_csv(f"{a.out}/citations.csv", cites)
    write_csv(f"{a.out}/evidence_hits.csv", hits)
    print(f"\nSaved to {a.out}/: grounding_summary.json, statements.csv, responses.csv, "
          f"citations.csv, evidence_hits.csv")
    print("\nNEXT: these numbers are not usable in the paper until a human-labelled sample "
          "confirms the judge agrees with human judgement. Label ~50 rows of statements.csv "
          "and report the agreement.")


if __name__ == "__main__":
    main()
