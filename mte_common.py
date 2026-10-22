"""
Shared helpers for the MedTrustEval evaluation scripts.

Everything here uses only the Python standard library.
"""
from __future__ import annotations

import json
import os
import re
from collections import OrderedDict

from evaluate_accuracy import (  # re-exported so every script scores answers the same way
    get_dataset,
    get_key,
    load_jsonl,
    normalize_answer,
    options_as_dict,
    wilson_ci,
)

__all__ = [
    "get_dataset", "get_key", "load_jsonl", "normalize_answer", "options_as_dict", "wilson_ci",
    "CITATION_RE", "strip_citations", "split_sentences", "get_reasoning", "get_docs",
    "sentence_citations", "is_correct", "get_confidence", "write_json", "write_csv", "pct",
]

CITATION_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def strip_citations(text: str) -> str:
    text = re.sub(r"\s+", " ", CITATION_RE.sub("", text))
    return re.sub(r"\s+([.,;:!?])", r"\1", text).strip()


_ABBREV = r"(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\bet al)(?<!\bDr)(?<!\bFig)"


def split_sentences(text: str) -> list[str]:
    """
    Split an explanation into sentence-level statements.
    A citation marker that follows the period ("... nerve. [2]") stays with its sentence.
    """
    if not text:
        return []
    text = re.sub(r"\s+", " ", str(text)).strip()
    # move "sentence. [1]" -> "sentence [1]."
    text = re.sub(r"([.!?])\s*((?:\[\d+(?:\s*[,;]\s*\d+)*\]\s*)+)", lambda m: " " + m.group(2).strip() + m.group(1), text)
    parts = re.split(_ABBREV + r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", text)
    out = []
    for p in parts:
        p = p.strip()
        if len(strip_citations(p)) >= 3:
            out.append(p)
    return out


def sentence_citations(sentence: str) -> list[int]:
    nums = []
    for grp in CITATION_RE.findall(sentence):
        nums.extend(int(n) for n in re.split(r"\s*[,;]\s*", grp) if n)
    return nums


_PLACEHOLDER = re.compile(r"^\s*one[- ]sentence explanation\s*[:.\-]?\s*", re.I)


def clean_reasoning(text: str) -> str:
    """Remove prompt-template text the model sometimes echoes (e.g. 'One sentence explanation:')."""
    return _PLACEHOLDER.sub("", text or "").strip()


def get_reasoning(row: dict) -> str:
    for f in ("reasoning", "explanation", "rationale"):
        v = row.get(f)
        if isinstance(v, str) and v.strip():
            return clean_reasoning(v)
    ma = row.get("model_answer")
    if isinstance(ma, dict):
        return clean_reasoning(str(ma.get("reasoning", "")))
    return ""


def _doc_text(d) -> str:
    if isinstance(d, str):
        return d
    if isinstance(d, dict):
        title = d.get("title") or ""
        body = d.get("content") or d.get("contents") or d.get("text") or d.get("snippet") or ""
        return (f"{title}. {body}" if title and body else (body or title)).strip()
    return str(d)


def get_docs(row: dict) -> list[str]:
    """Return retrieved documents as plain text, in the order they were numbered [1]..[k]."""
    for f in ("retrieved_docs", "retrieved_documents", "documents", "docs", "snippets", "contexts"):
        v = row.get(f)
        if isinstance(v, list) and v:
            return [_doc_text(d) for d in v]
        if isinstance(v, dict) and v:  # {"1": {...}, "2": {...}}
            return [_doc_text(v[k]) for k in sorted(v, key=lambda x: int(re.sub(r"\D", "", str(x)) or 0))]
    return []


def is_correct(row: dict) -> tuple[bool, bool]:
    """Returns (valid, correct)."""
    opts = row.get("options")
    gold = normalize_answer(row.get("correct_answer", row.get("answer")), opts)
    pred = normalize_answer(row.get("model_answer"), opts)
    return pred is not None, (pred is not None and pred == gold)


def get_confidence(row: dict) -> float | None:
    """Self-reported confidence rescaled to [0, 1]; None if missing/unparseable."""
    c = row.get("confidence")
    if c is None and isinstance(row.get("model_answer"), dict):
        c = row["model_answer"].get("confidence")
    if isinstance(c, str):
        m = re.search(r"-?\d+(?:\.\d+)?", c)
        c = float(m.group()) if m else None
    if c is None or isinstance(c, bool):
        return None
    c = float(c)
    if c > 1.0:
        c = c / 100.0
    if not 0.0 <= c <= 1.0:
        return None
    return c


def write_json(path: str, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def write_csv(path: str, rows: list[dict]):
    import csv
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = list(OrderedDict((k, None) for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})


def pct(x) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:.1f}%"
