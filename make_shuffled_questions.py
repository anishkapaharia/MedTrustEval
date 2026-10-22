#!/usr/bin/env python3
"""
Create a copy of the question file with the answer options shuffled, for the
consistency (position-bias) test.

Each output row gets:
  options        the shuffled options, relabelled A, B, C, ...
  correct_answer the correct letter in the shuffled order
  option_map     {shuffled_letter: original_letter}   <- the pipeline must keep this field in its log
  shuffle_seed   the seed used

The shuffle is guaranteed to move the correct answer to a different letter
(when there are 2+ options), so a model that always picks the same position
can't stay "correct" by accident.

Usage
    python make_shuffled_questions.py --in mirage_validation_50.jsonl --out mirage_validation_50_shuffled.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

from mte_common import load_jsonl, normalize_answer, options_as_dict

LETTERS = "ABCDEFGHIJ"


def shuffle_row(row: dict, rng: random.Random, seed: int) -> dict:
    opts = options_as_dict(row.get("options"))
    gold = normalize_answer(row.get("correct_answer", row.get("answer")), row.get("options"))
    orig_letters = list(opts)
    for _ in range(100):
        perm = orig_letters[:]
        rng.shuffle(perm)
        new_gold = LETTERS[perm.index(gold)] if gold in perm else None
        if len(perm) < 2 or new_gold != gold:
            break
    new = dict(row)
    new.pop("_line", None)
    new["options"] = {LETTERS[i]: opts[o] for i, o in enumerate(perm)}
    new["option_map"] = {LETTERS[i]: o for i, o in enumerate(perm)}
    if "correct_answer" in row or "answer" in row:
        new["correct_answer"] = new_gold
    new["shuffle_seed"] = seed
    return new


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    rows = load_jsonl(args.inp)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(shuffle_row(r, rng, args.seed)) + "\n")
    print(f"Wrote {len(rows)} shuffled questions to {args.out}")


if __name__ == "__main__":
    main()
