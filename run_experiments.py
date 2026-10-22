#!/usr/bin/env python3
"""
Run the MedTrustEval experiments yourself, using only Ollama (no MedRAG install needed).

How this works without MedRAG
-----------------------------
MIRAGE uses *question-only retrieval*: the 5 passages depend only on the question, not on the
answer options or on which run it is. So for the SAME questions we can reuse the 5 passages
that Justin's pipeline already retrieved (saved in rag_results.jsonl). That means:

  * repeat runs      -> same questions, same passages, new answers      (no MedRAG needed)
  * shuffled options -> same questions, same passages, options reordered (no MedRAG needed)
  * new prompt       -> same questions, same passages, new instructions  (no MedRAG needed)

Only NEW questions (e.g. expanding to 100-200) need new retrieval, i.e. Justin's MedRAG setup.

Settings are kept identical to the pilot: llama3:8b via Ollama, temperature 0.7, JSON output.

Setup (once)
  1. Install Ollama from https://ollama.com and open it.
  2. In Terminal:  ollama pull llama3:8b

Usage examples
  # both systems, new prompt, original option order, run 1
  python3 run_experiments.py --source logs/rag_results.jsonl --system both --out-dir logs/v2 --run 1

  # repeat runs 2..5
  for i in 2 3 4 5; do python3 run_experiments.py --source logs/rag_results.jsonl --system both --out-dir logs/v2/repeats --run $i; done

  # shuffled options
  python3 run_experiments.py --source logs/rag_results.jsonl --system both --out-dir logs/v2/shuffled --shuffle

The script can be stopped and restarted: questions already saved in the output file are skipped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

from make_shuffled_questions import shuffle_row
from mte_common import CITATION_RE, get_key, load_jsonl, split_sentences

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3:8b"
TEMPERATURE = 0.7
PROMPT_VERSION = "v2-2to4-sentences"

LLM_ONLY_PROMPT = """You are answering a medical multiple-choice question.

Question:
{question}

Options:
{options}

Choose the single best option. Explain your answer in 2-4 sentences.
Reply ONLY with a JSON object in exactly this format:
{{"answer": "<one option letter>", "confidence": <integer 0-100>, "reasoning": "<your 2-4 sentence explanation>"}}"""

RAG_PROMPT = """You are answering a medical multiple-choice question. Use the numbered documents below.

Documents:
{documents}

Question:
{question}

Options:
{options}

Choose the single best option. Explain your answer in 2-4 sentences, and end every sentence with the
number of the document that supports it in square brackets, like [1] or [3].
Reply ONLY with a JSON object in exactly this format:
{{"answer": "<one option letter>", "confidence": <integer 0-100>, "reasoning": "<your 2-4 sentence explanation with [n] citations>"}}"""


# ---------------------------------------------------------------- helpers
def fmt_options(opts: dict) -> str:
    return "\n".join(f"{k}. {v}" for k, v in opts.items())


def fmt_docs(docs: list) -> str:
    out = []
    for i, d in enumerate(docs, 1):
        if isinstance(d, dict):
            title = d.get("title", "")
            body = d.get("content") or d.get("contents") or d.get("text") or ""
            out.append(f"[{i}] {title}\n{body}".strip())
        else:
            out.append(f"[{i}] {d}")
    return "\n\n".join(out)


def call_ollama(prompt: str, url: str = OLLAMA_URL, model: str = MODEL, timeout: int = 600,
                threads: int | None = None, num_ctx: int | None = None, keep_alive: str | None = None) -> str:
    options = {"temperature": TEMPERATURE}
    if threads:
        options["num_thread"] = threads   # fewer CPU cores -> laptop stays responsive (slower runs)
    if num_ctx:
        options["num_ctx"] = num_ctx      # smaller context window -> less RAM (RAG prompts get truncated)
    payload = {"model": model, "prompt": prompt, "stream": False, "format": "json", "options": options}
    if keep_alive:
        payload["keep_alive"] = keep_alive   # e.g. "0" unloads the model from RAM after each answer
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())["response"]


def parse_response(text: str) -> dict:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        obj = json.loads(m.group()) if m else {}
    if not isinstance(obj, dict):
        obj = {}
    return obj


def parse_citations(reasoning: str, n_docs: int) -> tuple[list, bool]:
    """Same idea as the pilot pipeline: one entry per sentence; valid only if every sentence
    ends with a [n] marker (1..n_docs) before or right after its full stop."""
    out, valid = [], bool(reasoning.strip())
    for s in split_sentences(reasoning):
        nums = [int(x) for grp in CITATION_RE.findall(s) for x in re.split(r"\s*[,;]\s*", grp) if x]
        ends_with_marker = bool(re.search(r"\[\d+(?:\s*[,;]\s*\d+)*\]\s*[.!?]?\s*$", s))
        ok = ends_with_marker and nums and all(1 <= n <= n_docs for n in nums)
        valid = valid and bool(ok)
        out.append({"sentence": s, "citation": nums[0] if len(nums) == 1 else (nums or None)})
    return out, valid


def build_questions(source: str) -> list[dict]:
    """Questions + Justin's retrieved passages, taken from his RAG log (or any file with the same fields)."""
    rows = []
    for r in load_jsonl(source):
        rows.append({k: r.get(k) for k in ("dataset", "dataset_index", "id", "question", "options",
                                           "correct_answer", "retrieved_docs", "retrieval_scores")
                     if r.get(k) is not None})
    return rows


def already_done(path: str) -> set:
    if not os.path.exists(path):
        return set()
    return {get_key(r) for r in load_jsonl(path)}


# ---------------------------------------------------------------- main loop
def run(questions, system, out_path, run_no, shuffle, seed, url, model, max_retries=2, sleep=0.0,
        threads=None, num_ctx=None, keep_alive=None):
    done = already_done(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rng = random.Random(seed)
    n_new = n_fail = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for q in questions:
            q2 = shuffle_row(q, rng, seed) if shuffle else dict(q)
            if get_key(q2) in done:
                continue
            docs = q.get("retrieved_docs") or []
            if system == "rag" and not docs:
                print(f"  [skip] no saved passages for: {q['question'][:60]}", file=sys.stderr)
                continue
            prompt = (RAG_PROMPT.format(documents=fmt_docs(docs), question=q2["question"],
                                        options=fmt_options(q2["options"]))
                      if system == "rag" else
                      LLM_ONLY_PROMPT.format(question=q2["question"], options=fmt_options(q2["options"])))
            raw, obj = "", {}
            for attempt in range(max_retries + 1):
                try:
                    raw = call_ollama(prompt, url, model, threads=threads, num_ctx=num_ctx,
                                      keep_alive=keep_alive)
                    obj = parse_response(raw)
                    if obj.get("answer"):
                        break
                except (urllib.error.URLError, ConnectionError) as e:
                    sys.exit(f"Can't reach Ollama at {url}. Is the Ollama app open? ({e})")
                except Exception as e:  # malformed JSON etc. -> retry
                    raw = raw or str(e)
            if not obj.get("answer"):
                n_fail += 1
            reasoning = str(obj.get("reasoning", ""))
            rec = {
                "dataset": q2.get("dataset"), "dataset_index": q2.get("dataset_index"),
                "question": q2["question"], "options": q2["options"], "correct_answer": q2.get("correct_answer"),
                "model_answer": obj.get("answer"), "confidence": obj.get("confidence"), "reasoning": reasoning,
                "raw_response": raw,
                "temperature": TEMPERATURE, "model": model, "system": system,
                "prompt_version": PROMPT_VERSION, "run": run_no,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            if shuffle:
                rec["option_map"] = q2["option_map"]
                rec["shuffle_seed"] = seed
            if system == "rag":
                cits, valid = parse_citations(reasoning, len(docs))
                rec.update({"retrieved_docs": docs, "retrieval_scores": q.get("retrieval_scores"),
                            "citations": cits, "citations_valid": valid})
            else:
                rec.update({"retrieved_docs": None, "citations_valid": None})
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_new += 1
            print(f"  {system} run {run_no}{' shuffled' if shuffle else ''}: {n_new} done", end="\r")
            if sleep:
                time.sleep(sleep)
    print(f"\n  {system}: {n_new} new answers saved to {out_path}"
          + (f" ({n_fail} with no readable answer)" if n_fail else ""))
    return n_new, n_fail


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="logs/rag_results.jsonl",
                    help="Justin's RAG log: provides the questions AND the saved passages")
    ap.add_argument("--system", choices=["llm_only", "rag", "both"], default="both")
    ap.add_argument("--out-dir", default="logs/v2")
    ap.add_argument("--run", type=int, default=1, help="run number (used in file names)")
    ap.add_argument("--shuffle", action="store_true", help="shuffle answer options (position-bias test)")
    ap.add_argument("--seed", type=int, default=13, help="shuffle seed (keep the same for both systems)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (for a quick test)")
    ap.add_argument("--url", default=OLLAMA_URL)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU cores Ollama may use (e.g. 4). Fewer cores = slower run, but the laptop stays usable.")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds to pause between questions (e.g. 1) to keep the laptop cooler")
    ap.add_argument("--num-ctx", type=int, default=None,
                    help="context window, e.g. 4096. Smaller = less RAM. Long RAG prompts are truncated.")
    ap.add_argument("--keep-alive", default=None,
                    help='how long Ollama keeps the model in RAM, e.g. "0" to unload after each answer, "10m" to keep it')
    a = ap.parse_args(argv)

    qs = build_questions(a.source)[: a.limit] if a.limit else build_questions(a.source)
    print(f"{len(qs)} questions from {a.source}; model {a.model}; temperature {TEMPERATURE}; prompt {PROMPT_VERSION}")
    systems = ["llm_only", "rag"] if a.system == "both" else [a.system]
    for s in systems:
        name = f"{s}_shuffled.jsonl" if a.shuffle else f"{s}_run{a.run}.jsonl"
        run(qs, s, os.path.join(a.out_dir, name), a.run, a.shuffle, a.seed, a.url, a.model,
            sleep=a.sleep, threads=a.threads, num_ctx=a.num_ctx, keep_alive=a.keep_alive)


if __name__ == "__main__":
    main()
