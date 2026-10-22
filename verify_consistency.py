#!/usr/bin/env python3
"""Independent consistency analysis across repeat runs 1-3 (llama3.2:3b, prompt v2)."""
import json, itertools, math, random
from collections import Counter, defaultdict

B = "/home/claude/eval/real/logs/v2_3b/"
L = lambda n: [json.loads(l) for l in open(B + n) if l.strip()]
k = lambda r: (r["dataset"], r["dataset_index"])
def pr(r):
    a = str(r.get("model_answer") or "").strip().upper()
    return a[0] if a[:1] in "ABCD" else None
ok = lambda r: pr(r) == str(r["correct_answer"]).strip().upper()

SYS = {"LLM-only": ["llm_only_run1.jsonl", "llm_only_run2.jsonl", "llm_only_run3.jsonl"],
       "LLM+RAG":  ["rag_run1.jsonl", "rag_run2.jsonl", "rag_run3.jsonl"]}

print("=" * 74)
print("SANITY: are the three runs actually different runs?")
print("=" * 74)
for name, files in SYS.items():
    runs = [L(f) for f in files]
    for f, rows in zip(files, runs):
        ts = sorted(r["timestamp"] for r in rows)
        print(f"  {f:24s} n={len(rows)} run_field={Counter(r.get('run') for r in rows)} "
              f"first={ts[0][:19]} last={ts[-1][:19]}")
    # identical text would mean a copied file
    txt = [tuple(sorted((k(r), r.get("reasoning", "")) for r in rows)) for rows in runs]
    for (i, j) in itertools.combinations(range(3), 2):
        print(f"    run{i+1} vs run{j+1}: reasoning text identical? {txt[i] == txt[j]}")

def wilson(x, n, z=1.959963984540054):
    if n == 0: return (0, 0)
    p = x/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0, c-h), min(1, c+h)

def fleiss_kappa(table):
    """table: list of per-item Counters of category -> count, equal n raters"""
    N = len(table); n = sum(table[0].values())
    cats = sorted({c for t in table for c in t})
    p_j = {c: sum(t.get(c, 0) for t in table)/(N*n) for c in cats}
    P_i = [(sum(v*v for v in t.values()) - n)/(n*(n-1)) for t in table]
    Pbar = sum(P_i)/N
    Pe = sum(v*v for v in p_j.values())
    return (Pbar - Pe)/(1 - Pe) if Pe < 1 else float("nan")

print("\n" + "=" * 74)
print("1. ANSWER AGREEMENT ACROSS THE THREE REPEAT RUNS")
print("=" * 74)
store = {}
for name, files in SYS.items():
    runs = [{k(r): r for r in L(f)} for f in files]
    common = sorted(set(runs[0]) & set(runs[1]) & set(runs[2]))
    unanimous = 0; pair_agree = 0; pair_total = 0
    table = []; majority_correct = 0; flip_corr = 0
    per_run_acc = [0, 0, 0]
    changed_correctness = 0
    for key in common:
        ans = [pr(runs[i][key]) for i in range(3)]
        cor = [ok(runs[i][key]) for i in range(3)]
        for i in range(3):
            per_run_acc[i] += cor[i]
        c = Counter(ans); table.append(c)
        if len(c) == 1: unanimous += 1
        for a, b in itertools.combinations(ans, 2):
            pair_total += 1; pair_agree += (a == b)
        maj = c.most_common(1)[0][0]
        if maj == str(runs[0][key]["correct_answer"]).strip().upper(): majority_correct += 1
        if len(set(cor)) > 1: changed_correctness += 1
    n = len(common)
    lo, hi = wilson(unanimous, n)
    print(f"\n{name}  (n = {n} questions, 3 runs each)")
    print(f"  identical answer in all 3 runs : {unanimous}/{n} = {100*unanimous/n:.1f}%  "
          f"(95% CI {100*lo:.1f}–{100*hi:.1f})")
    print(f"  pairwise agreement             : {pair_agree}/{pair_total} = {100*pair_agree/pair_total:.1f}%")
    print(f"  Fleiss' kappa across 3 runs    : {fleiss_kappa(table):.3f}")
    print(f"  accuracy per run               : " +
          "  ".join(f"run{i+1} {100*per_run_acc[i]/n:.1f}%" for i in range(3)) +
          f"   (spread {100*(max(per_run_acc)-min(per_run_acc))/n:.1f} pts)")
    print(f"  majority-vote accuracy         : {100*majority_correct/n:.1f}%")
    print(f"  questions where the answer flips between right and wrong across runs: "
          f"{changed_correctness}/{n} = {100*changed_correctness/n:.1f}%")
    store[name] = dict(n=n, unan=unanimous, pair=pair_agree/pair_total,
                       kappa=fleiss_kappa(table), accs=[a/n for a in per_run_acc],
                       maj=majority_correct/n, flip=changed_correctness/n, runs=runs, common=common)

print("\n" + "=" * 74)
print("2. HOW MANY DISTINCT ANSWERS DOES A QUESTION GET?")
print("=" * 74)
for name in SYS:
    s = store[name]
    d = Counter(len(set(pr(s["runs"][i][key]) for i in range(3))) for key in s["common"])
    print(f"  {name:9s} 1 distinct answer: {d.get(1,0)},  2 distinct: {d.get(2,0)},  3 distinct: {d.get(3,0)}")

print("\n" + "=" * 74)
print("3. PER-DATASET UNANIMITY")
print("=" * 74)
for name in SYS:
    s = store[name]
    by = defaultdict(lambda: [0, 0])
    for key in s["common"]:
        ds = key[0]; by[ds][0] += 1
        if len(set(pr(s["runs"][i][key]) for i in range(3))) == 1: by[ds][1] += 1
    print(f"  {name}: " + "  ".join(f"{ds} {100*v[1]/v[0]:.0f}%" for ds, v in sorted(by.items())))

print("\n" + "=" * 74)
print("4. CONFIDENCE STABILITY ACROSS RUNS")
print("=" * 74)
for name in SYS:
    s = store[name]
    same_conf = 0; spreads = []
    for key in s["common"]:
        cs = [s["runs"][i][key].get("confidence") for i in range(3)]
        cs = [float(c) for c in cs if c is not None]
        if len(set(cs)) == 1: same_conf += 1
        if cs: spreads.append(max(cs) - min(cs))
    n = s["n"]
    print(f"  {name:9s} identical confidence in all 3 runs: {same_conf}/{n} = {100*same_conf/n:.0f}%   "
          f"mean spread (max−min) {sum(spreads)/len(spreads):.1f} points")

print("\n" + "=" * 74)
print("5. TWO KINDS OF INSTABILITY, SIDE BY SIDE")
print("=" * 74)
# shuffle stability recomputed here for comparison
for name, orig, shuf in [("LLM-only", "llm_only_run1.jsonl", "llm_only_shuffled.jsonl"),
                         ("LLM+RAG", "rag_run1.jsonl", "rag_shuffled.jsonl")]:
    o = {k(r): r for r in L(orig)}; s = {k(r): r for r in L(shuf)}
    common = sorted(set(o) & set(s)); same = 0
    for key in common:
        m = s[key].get("option_map") or {}
        if m.get(pr(s[key])) == pr(o[key]): same += 1
    st = store[name]
    print(f"  {name:9s} same answer on a plain repeat (3 runs unanimous): {100*st['unan']/st['n']:.1f}%   |   "
          f"same answer after reordering the options: {100*same/len(common):.1f}%")

print("\n" + "=" * 74)
print("6. DOES ACCURACY ITSELF MOVE BETWEEN RUNS?")
print("=" * 74)
for name in SYS:
    a = store[name]["accs"]
    print(f"  {name:9s} " + "  ".join(f"{100*x:.1f}%" for x in a) +
          f"   mean {100*sum(a)/3:.1f}%   range {100*(max(a)-min(a)):.1f} pts")
la = store["LLM-only"]["accs"]; ra = store["LLM+RAG"]["accs"]
print(f"\n  RAG − LLM-only difference in each run: " +
      "  ".join(f"run{i+1} {100*(ra[i]-la[i]):+.1f} pts" for i in range(3)))
