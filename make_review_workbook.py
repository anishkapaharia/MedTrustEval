#!/usr/bin/env python3
"""
Build the Day 9 manual-review workbook (one Excel file, four tabs) straight from the two logs.
No NLI model or PyTorch needed.

Tabs
  Instructions  what to fill in and the label definitions
  Examples      paired LLM-only vs RAG examples: all "RAG hurt" and "RAG helped" questions,
                plus a sample of "both wrong" and "both correct". Fill in: problem_source, notes
  Retrieval     sampled questions x 5 retrieved passages. Each rater fills in 1 (relevant) / 0 (not)
  Statements    sampled explanation sentences, each with the 5 passages. Each rater fills in S / C / N
  Scores        formulas: Precision@5, % agreement and Cohen's kappa (updates as you type)

Usage
  python make_review_workbook.py --llm logs/llm_only_results.jsonl --rag logs/rag_results.jsonl \
      --out review/MedTrustEval_Day9_Review.xlsx
"""
from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from mte_common import (get_docs, get_key, get_reasoning, is_correct, load_jsonl, split_sentences,
                        strip_citations, get_dataset)

FONT = "Arial"
HEAD = PatternFill("solid", start_color="1F3864")
INPUT = PatternFill("solid", start_color="FFF2CC")   # yellow = cells to fill in
WRAP = Alignment(wrap_text=True, vertical="top")


def style_header(ws, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF")
        cell.fill = HEAD
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.freeze_panes = "A2"


def body_font(ws):
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            if cell.font is None or not cell.font.bold:
                cell.font = Font(name=FONT, size=10)
            cell.alignment = WRAP


def fmt_opts(opts):
    return "\n".join(f"{k}. {v}" for k, v in (opts or {}).items())


def build(llm_path, rag_path, out, n_both=5, n_retrieval_per_ds=2, n_stmt_llm=20, n_stmt_rag=40, seed=7):
    rng = random.Random(seed)
    llm = {get_key(r): r for r in load_jsonl(llm_path)}
    rag = {get_key(r): r for r in load_jsonl(rag_path)}
    keys = [k for k in llm if k in rag]

    groups = defaultdict(list)
    for k in keys:
        a, b = is_correct(llm[k])[1], is_correct(rag[k])[1]
        g = "both_correct" if a and b else "rag_helped" if b else "rag_hurt" if a else "both_wrong"
        groups[g].append(k)

    wb = Workbook()
    # ---------------- Instructions
    ws = wb.active
    ws.title = "Instructions"
    lines = [
        ("MedTrustEval: Day 9 manual review", True),
        ("Yellow cells are the only cells to fill in. Everything else comes from the logs.", False),
        ("", False),
        ("Examples tab: read each pair and fill in problem_source and notes.", True),
        ("  retrieval: the passages don't contain what's needed", False),
        ("  evidence_use: the passages were good but the model ignored or misread them", False),
        ("  citation: answer is fine but citations are missing or point to the wrong passage", False),
        ("  position_bias: the model seems to pick A regardless of content", False),
        ("  question: the question or answer key is ambiguous or wrong", False),
        ("  other: explain in notes", False),
        ("", False),
        ("Retrieval tab: each rater, WITHOUT looking at the other's column, types 1 or 0.", True),
        ("  1 = the passage contains information useful for answering the question; 0 = it doesn't", False),
        ("", False),
        ("Statements tab: each rater, separately, types S, C or N.", True),
        ("  S (supported): at least one of the 5 passages states or clearly implies the sentence", False),
        ("  C (contradicted): a passage clearly says the opposite (this counts as a hallucination)", False),
        ("  N (not addressed): the passages don't say either way (NOT counted as a hallucination)", False),
        ("  For LLM-only sentences, the passages are the ones retrieved for RAG. The LLM-only model never saw them.", False),
        ("", False),
        ("Scores tab updates automatically: Precision@5, % agreement and Cohen's kappa.", True),
        ("  Kappa of 0.6 or higher = good agreement. Below 0.4 = discuss the definitions and relabel.", False),
        ("", False),
        ("Example row (Statements tab): 'Bell's palsy affects the facial nerve' + a passage saying so -> S", False),
    ]
    for i, (t, bold) in enumerate(lines, 1):
        c = ws.cell(row=i, column=1, value=t)
        c.font = Font(name=FONT, bold=bold, size=12 if i == 1 else 10)
    ws.column_dimensions["A"].width = 110

    # ---------------- Examples
    ws = wb.create_sheet("Examples")
    cols = ["group", "dataset", "question", "options", "correct", "llm_answer", "llm_conf", "llm_explanation",
            "rag_answer", "rag_conf", "rag_explanation", "doc1", "doc2", "doc3", "doc4", "doc5",
            "problem_source", "notes"]
    ws.append(cols)
    order = [("rag_hurt", None), ("rag_helped", None), ("both_wrong", n_both), ("both_correct", n_both)]
    for g, n in order:
        ks = groups.get(g, [])
        ks = ks if n is None else rng.sample(ks, min(n, len(ks)))
        for k in ks:
            l, r = llm[k], rag[k]
            docs = get_docs(r) + [""] * 5
            ws.append([g, get_dataset(r), r.get("question"), fmt_opts(r.get("options")), r.get("correct_answer"),
                       l.get("model_answer"), l.get("confidence"), get_reasoning(l),
                       r.get("model_answer"), r.get("confidence"), get_reasoning(r),
                       *[d[:1500] for d in docs[:5]], "", ""])
    style_header(ws, len(cols))
    body_font(ws)
    widths = [12, 10, 50, 35, 8, 8, 8, 45, 8, 8, 55, 45, 45, 45, 45, 45, 16, 35]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = w
    dv = DataValidation(type="list", formula1='"retrieval,evidence_use,citation,position_bias,question,other"',
                        allow_blank=True)
    ws.add_data_validation(dv)
    for r in range(2, ws.max_row + 1):
        for c in (17, 18):
            ws.cell(row=r, column=c).fill = INPUT
        dv.add(f"Q{r}")
    n_examples = ws.max_row - 1

    # ---------------- Retrieval
    ws = wb.create_sheet("Retrieval")
    ws.append(["question_no", "dataset", "question", "doc_rank", "passage", "rater1 (1/0)", "rater2 (1/0)"])
    by_ds = defaultdict(list)
    for k in keys:
        by_ds[get_dataset(rag[k])].append(k)
    qno = 0
    for ds in sorted(by_ds):
        for k in rng.sample(by_ds[ds], min(n_retrieval_per_ds, len(by_ds[ds]))):
            qno += 1
            for i, d in enumerate(get_docs(rag[k])[:5], 1):
                ws.append([qno, ds, rag[k].get("question") if i == 1 else "(same as above)", i, d[:1500], None, None])
    style_header(ws, 7)
    body_font(ws)
    for col, w in zip("ABCDEFG", [10, 10, 50, 8, 90, 12, 12]):
        ws.column_dimensions[col].width = w
    dv2 = DataValidation(type="whole", operator="between", formula1="0", formula2="1", allow_blank=True)
    ws.add_data_validation(dv2)
    n_ret = ws.max_row - 1
    for r in range(2, ws.max_row + 1):
        for c in "FG":
            ws[f"{c}{r}"].fill = INPUT
            dv2.add(f"{c}{r}")

    # ---------------- Statements
    ws = wb.create_sheet("Statements")
    ws.append(["item", "system", "dataset", "question", "statement", "doc1", "doc2", "doc3", "doc4", "doc5",
               "rater1 (S/C/N)", "rater2 (S/C/N)"])
    pool = {"llm_only": [], "rag": []}
    for k in keys:
        docs = get_docs(rag[k])
        for sysname, row in (("llm_only", llm[k]), ("rag", rag[k])):
            for s in split_sentences(get_reasoning(row)):
                txt = strip_citations(s)
                if "one sentence explanation" in txt.lower() and len(txt) < 40:
                    continue   # prompt placeholder, not a statement
                pool[sysname].append((sysname, k, txt, docs))
    picked = rng.sample(pool["llm_only"], min(n_stmt_llm, len(pool["llm_only"]))) + \
        rng.sample(pool["rag"], min(n_stmt_rag, len(pool["rag"])))
    rng.shuffle(picked)   # mix systems so raters don't label by system
    for i, (sysname, k, txt, docs) in enumerate(picked, 1):
        d = docs + [""] * 5
        ws.append([i, sysname, get_dataset(rag[k]), rag[k].get("question"), txt, *[x[:1200] for x in d[:5]], None, None])
    style_header(ws, 12)
    body_font(ws)
    for col, w in zip("ABCDEFGHIJKL", [6, 9, 10, 40, 50, 40, 40, 40, 40, 40, 12, 12]):
        ws.column_dimensions[col].width = w
    dv3 = DataValidation(type="list", formula1='"S,C,N"', allow_blank=True)
    ws.add_data_validation(dv3)
    n_st = ws.max_row - 1
    for r in range(2, ws.max_row + 1):
        for c in "KL":
            ws[f"{c}{r}"].fill = INPUT
            dv3.add(f"{c}{r}")

    # ---------------- Scores (formulas only)
    ws = wb.create_sheet("Scores")
    R = f"Retrieval!F2:F{n_ret + 1}"
    R2 = f"Retrieval!G2:G{n_ret + 1}"
    S1 = f"Statements!K2:K{n_st + 1}"
    S2 = f"Statements!L2:L{n_st + 1}"
    rows = [
        ("MedTrustEval: manual review scores (update automatically)", None, None),
        ("Retrieval: Precision@5", None, None),
        ("Rater 1 labelled", f"=COUNT({R})", "passages"),
        ("Rater 2 labelled", f"=COUNT({R2})", "passages"),
        ("Precision@5, rater 1", f'=IF(B3=0,"",AVERAGE({R}))', "share of passages marked relevant"),
        ("Precision@5, rater 2", f'=IF(B4=0,"",AVERAGE({R2}))', ""),
        ("Precision@5, average of raters", '=IF(OR(B5="",B6=""),"",AVERAGE(B5,B6))', "goes in the paper"),
        ("Both labelled", f'=SUMPRODUCT(ISNUMBER({R})*ISNUMBER({R2}))', ""),
        ("% agreement", f'=IF(B8=0,"",SUMPRODUCT(ISNUMBER({R})*ISNUMBER({R2})*({R}={R2}))/B8)', ""),
        ("p(1) rater 1 | p(1) rater 2", f'=IF(B8=0,"",SUMPRODUCT(ISNUMBER({R2})*({R}=1))/B8)',
         f'=IF(B8=0,"",SUMPRODUCT(ISNUMBER({R})*({R2}=1))/B8)'),
        ("Cohen's kappa", '=IF(B8=0,"",IF(B10*C10+(1-B10)*(1-C10)=1,"",(B9-(B10*C10+(1-B10)*(1-C10)))/(1-(B10*C10+(1-B10)*(1-C10)))))',
         "0.6+ good; below 0.4 discuss and relabel"),
        ("", None, None),
        ("Statements: S / C / N", None, None),
        ("Both labelled", f'=SUMPRODUCT(({S1}<>"")*({S2}<>""))', "statements"),
        ("% agreement", f'=IF(B14=0,"",SUMPRODUCT(({S1}<>"")*({S1}={S2}))/B14)', ""),
    ]
    for lab in ("S", "C", "N"):
        rows.append((f"share labelled {lab} (rater 1 | rater 2)",
                     f'=IF(B14=0,"",SUMPRODUCT(({S2}<>"")*({S1}="{lab}"))/B14)',
                     f'=IF(B14=0,"",SUMPRODUCT(({S1}<>"")*({S2}="{lab}"))/B14)'))
    rows.append(("Expected agreement by chance", '=IF(B14=0,"",B16*C16+B17*C17+B18*C18)', ""))
    rows.append(("Cohen's kappa", '=IF(B14=0,"",IF(B19=1,"",(B15-B19)/(1-B19)))', "0.6+ good; below 0.4 discuss and relabel"))
    rows.append(("", None, None))
    rows.append(("Hallucination rate from human labels (rater 1)", f'=IF(COUNTA({S1})=0,"",COUNTIF({S1},"C")/COUNTA({S1}))',
                 "share of labelled sentences marked C"))
    rows.append(("Examples reviewed", f'=COUNTA(Examples!Q2:Q{n_examples + 1})', f"of {n_examples}"))
    for i, (a, b, c) in enumerate(rows, 1):
        ws.cell(row=i, column=1, value=a).font = Font(name=FONT, bold=(b is None and a != ""), size=10)
        if b is not None:
            ws.cell(row=i, column=2, value=b).font = Font(name=FONT, size=10)
        if c is not None:
            ws.cell(row=i, column=3, value=c).font = Font(name=FONT, size=10, italic=not str(c).startswith("="))
    for r in (5, 6, 7, 9, 10, 15, 16, 17, 18, 19, 22):
        ws.cell(row=r, column=2).number_format = "0.0%"
    for r in (10, 16, 17, 18):
        ws.cell(row=r, column=3).number_format = "0.0%"
    ws.cell(row=11, column=2).number_format = "0.00"
    ws.cell(row=20, column=2).number_format = "0.00"
    ws.column_dimensions["A"].width = 44
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 40

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    wb.save(out)
    return {"examples": n_examples, "retrieval_rows": n_ret, "statements": n_st,
            "groups": {g: len(v) for g, v in groups.items()},
            "statement_pool": {k: len(v) for k, v in pool.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="logs/llm_only_results.jsonl")
    ap.add_argument("--rag", default="logs/rag_results.jsonl")
    ap.add_argument("--out", default="review/MedTrustEval_Day9_Review.xlsx")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    info = build(a.llm, a.rag, a.out, seed=a.seed)
    print(f"Wrote {a.out}: {info}")


if __name__ == "__main__":
    main()
