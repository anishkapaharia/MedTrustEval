#!/usr/bin/env python3
"""Independent re-derivation of the llama3.2:3b numbers. Written from scratch:
no imports from the project's eval modules."""
import json, math, random
from collections import Counter, defaultdict

BASE = "/home/claude/eval/real/logs/v2_3b/"

def load(name):
    return [json.loads(l) for l in open(BASE + name) if l.strip()]

def key(r):
    return (r["dataset"], r["dataset_index"])

def pred(r):
    a = r.get("model_answer")
    if a is None:
        return None
    a = str(a).strip().upper()
    return a[0] if a and a[0] in "ABCD" else None

def correct(r):
    p = pred(r)
    return None if p is None else p == str(r["correct_answer"]).strip().upper()

# ---------- statistics, implemented here from scratch ----------
def wilson(k, n, z=1.959963984540054):
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c-h), min(1.0, c+h))

def binom_pmf(k, n, p=0.5):
    return math.comb(n, k) * p**n

def mcnemar_exact(b, c):
    """two-sided exact binomial test on the discordant pairs"""
    n = b + c
    if n == 0: return 1.0
    obs = min(b, c)
    tail = sum(binom_pmf(i, n) for i in range(0, obs+1))
    return min(1.0, 2*tail)

def boot_diff_ci(pairs, reps=10000, seed=42):
    rng = random.Random(seed)
    n = len(pairs)
    idx = range(n)
    diffs = []
    for _ in range(reps):
        s = [pairs[rng.randrange(n)] for _ in idx]
        diffs.append(sum(y for _, y in s)/n - sum(x for x, _ in s)/n)
    diffs.sort()
    lo = diffs[int(0.025*reps)]
    hi = diffs[int(0.975*reps)-1]
    return lo*100, hi*100

def report_pair(name_a, rows_a, name_b, rows_b):
    a = {key(r): r for r in rows_a}
    b = {key(r): r for r in rows_b}
    common = sorted(set(a) & set(b))
    pairs = []
    both = ah = bh = neither = 0
    for k in common:
        ca, cb = correct(a[k]), correct(b[k])
        pairs.append((1 if ca else 0, 1 if cb else 0))
        if ca and cb: both += 1
        elif cb and not ca: bh += 1
        elif ca and not cb: ah += 1
        else: neither += 1
    ka = sum(x for x, _ in pairs); kb = sum(y for _, y in pairs); n = len(pairs)
    la, ha = wilson(ka, n); lb, hb = wilson(kb, n)
    p = mcnemar_exact(ah, bh)   # ah = a-only-correct, bh = b-only-correct
    lo, hi = boot_diff_ci(pairs)
    print(f"\n### {name_a} vs {name_b}  (n paired = {n})")
    print(f"  {name_a:22s} {ka}/{n} = {100*ka/n:.1f}%  (95% CI {100*la:.1f}–{100*ha:.1f})")
    print(f"  {name_b:22s} {kb}/{n} = {100*kb/n:.1f}%  (95% CI {100*lb:.1f}–{100*hb:.1f})")
    print(f"  difference ({name_b} − {name_a}) = {100*(kb-ka)/n:+.1f} pts  "
          f"(bootstrap 95% CI {lo:+.1f} to {hi:+.1f})")
    print(f"  paired outcomes: both correct {both}, only {name_b} correct {bh}, "
          f"only {name_a} correct {ah}, both wrong {neither}")
    print(f"  exact McNemar on {ah}/{bh} discordant pairs: p = {p:.4f}")
    return dict(n=n, ka=ka, kb=kb, both=both, b_only=bh, a_only=ah, neither=neither, p=p)

# ---------- load ----------
lo1  = load("llm_only_run1.jsonl")
rag1 = load("rag_run1.jsonl")
los  = load("llm_only_shuffled.jsonl")
rags = load("rag_shuffled.jsonl")

print("=" * 72)
print("SANITY CHECKS")
print("=" * 72)
for nm, rows in [("llm_only run1", lo1), ("rag run1", rag1),
                 ("llm_only shuffled", los), ("rag shuffled", rags)]:
    models = Counter(r.get("model") for r in rows)
    temps  = Counter(r.get("temperature") for r in rows)
    pv     = Counter(r.get("prompt_version") for r in rows)
    invalid = sum(1 for r in rows if pred(r) is None)
    dupes = len(rows) - len(set(key(r) for r in rows))
    print(f"{nm:20s} n={len(rows)} model={dict(models)} temp={dict(temps)} "
          f"prompt={dict(pv)} invalid={invalid} duplicate_keys={dupes}")

# confirm the shuffle really moved the gold letter, and that option TEXT is preserved
print("\nShuffle integrity (gold letter must move; option text set must be identical):")
for nm, orig, shuf in [("llm_only", lo1, los), ("rag", rag1, rags)]:
    o = {key(r): r for r in orig}; s = {key(r): r for r in shuf}
    common = sorted(set(o) & set(s))
    moved = same_text = map_ok = 0
    for k in common:
        if o[k]["correct_answer"] != s[k]["correct_answer"]:
            moved += 1
        if sorted(o[k]["options"].values()) == sorted(s[k]["options"].values()):
            same_text += 1
        m = s[k].get("option_map") or {}
        # option_map: new letter -> original letter
        if all(s[k]["options"][new] == o[k]["options"].get(old) for new, old in m.items()):
            map_ok += 1
    print(f"  {nm:9s} common={len(common)} gold letter moved {moved}/{len(common)}, "
          f"same option text {same_text}/{len(common)}, option_map consistent {map_ok}/{len(common)}")

print("\n" + "=" * 72)
print("1. ACCURACY — original option order (reproduces Table 3)")
print("=" * 72)
report_pair("LLM-only", lo1, "LLM+RAG", rag1)

print("\nLetter distributions (original order):")
print("  LLM-only predicted:", dict(sorted(Counter(pred(r) for r in lo1).items(), key=lambda x: str(x[0]))))
print("  RAG      predicted:", dict(sorted(Counter(pred(r) for r in rag1).items(), key=lambda x: str(x[0]))))
print("  gold             :", dict(sorted(Counter(r["correct_answer"] for r in lo1).items())))

print("\nPer dataset (original order):")
by = defaultdict(lambda: [0, 0, 0])
r1 = {key(r): r for r in rag1}
for r in lo1:
    d = r["dataset"]; by[d][0] += 1
    by[d][1] += 1 if correct(r) else 0
    o = r1.get(key(r))
    by[d][2] += 1 if (o and correct(o)) else 0
for d in sorted(by):
    n, a, b = by[d]
    print(f"  {d:10s} n={n}  LLM-only {100*a/n:5.1f}%   RAG {100*b/n:5.1f}%   diff {100*(b-a)/n:+5.1f} pts")

print("\n" + "=" * 72)
print("2. SHUFFLE TEST — does reordering the options change the answer?")
print("=" * 72)
for nm, orig, shuf in [("LLM-only", lo1, los), ("RAG", rag1, rags)]:
    o = {key(r): r for r in orig}; s = {key(r): r for r in shuf}
    common = sorted(set(o) & set(s))
    same_content = same_letter = 0
    for k in common:
        m = s[k].get("option_map") or {}      # new letter -> original letter
        ps, po = pred(s[k]), pred(o[k])
        if ps is not None and po is not None:
            if m.get(ps) == po:
                same_content += 1             # picked the same OPTION TEXT
            if ps == po:
                same_letter += 1              # picked the same LETTER
    n = len(common)
    print(f"  {nm:9s} stability (same option text chosen) {same_content}/{n} = {100*same_content/n:.1f}%   "
          f"same-letter rate {same_letter}/{n} = {100*same_letter/n:.1f}%")

print("\nAccuracy under shuffling:")
res_shuf = {}
for nm, orig, shuf in [("LLM-only", lo1, los), ("RAG", rag1, rags)]:
    res_shuf[nm] = report_pair(f"{nm} orig", orig, f"{nm} shuffled", shuf)

print("\nLetter distributions (shuffled order):")
print("  LLM-only predicted:", dict(sorted(Counter(pred(r) for r in los).items(), key=lambda x: str(x[0]))))
print("  RAG      predicted:", dict(sorted(Counter(pred(r) for r in rags).items(), key=lambda x: str(x[0]))))
print("  gold (shuffled)   :", dict(sorted(Counter(r["correct_answer"] for r in los).items())))

print("\n" + "=" * 72)
print("3. LLM-only vs RAG, both under shuffling")
print("=" * 72)
report_pair("LLM-only shuffled", los, "RAG shuffled", rags)

print("\n" + "=" * 72)
print("4. 'ALWAYS THE SAME LETTER' BASELINES")
print("=" * 72)
for nm, rows in [("orig", lo1), ("shuffled", los)]:
    g = Counter(r["correct_answer"] for r in rows)
    best = max(g.values())
    print(f"  {nm:9s}: most common gold letter = {max(g, key=g.get)} "
          f"-> always-that-letter baseline {100*best/len(rows):.1f}%")

print("\n" + "=" * 72)
print("5. CALIBRATION (ECE 10 bins, Brier) — recomputed independently")
print("=" * 72)
def calib(rows, label):
    xs = []
    for r in rows:
        c = r.get("confidence")
        if c is None: continue
        c = float(c)
        if c > 1: c /= 100.0
        c = min(max(c, 0.0), 1.0)
        xs.append((c, 1 if correct(r) else 0))
    n = len(xs)
    brier = sum((c - y) ** 2 for c, y in xs) / n
    ece = 0.0
    for i in range(10):
        lo_, hi_ = i/10, (i+1)/10
        b = [x for x in xs if (lo_ < x[0] <= hi_) or (i == 0 and x[0] == 0)]
        if not b: continue
        conf = sum(c for c, _ in b)/len(b); acc = sum(y for _, y in b)/len(b)
        ece += len(b)/n * abs(conf - acc)
    accs = [c for c, y in xs if y == 1]; wrongs = [c for c, y in xs if y == 0]
    gap = (sum(accs)/len(accs) - sum(wrongs)/len(wrongs)) * 100 if accs and wrongs else float("nan")
    conf_err = sum(1 for c, y in xs if y == 0 and c >= 0.8) / n * 100
    vals = Counter(round(c*100) for c, _ in xs)
    print(f"  {label:20s} n={n} mean confidence {100*sum(c for c,_ in xs)/n:.1f}%  accuracy {100*sum(y for _,y in xs)/n:.1f}%")
    print(f"  {'':20s} ECE {ece:.3f}  Brier {brier:.3f}  confidence gap (correct−wrong) {gap:+.1f} pts  "
          f"confident-error rate {conf_err:.1f}%")
    print(f"  {'':20s} distinct confidence values: {dict(sorted(vals.items()))}")
for nm, rows in [("LLM-only orig", lo1), ("RAG orig", rag1),
                 ("LLM-only shuffled", los), ("RAG shuffled", rags)]:
    calib(rows, nm)

print("\n" + "=" * 72)
print("6. CITATIONS (RAG only) — strict vs lenient")
print("=" * 72)
import re
CIT = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (t or "").strip()) if s.strip()]
for nm, rows in [("RAG orig", rag1), ("RAG shuffled", rags)]:
    strict = lenient = 0; total_sent = cited_sent = bad_num = 0
    for r in rows:
        ndocs = len(r.get("retrieved_docs") or [])
        ss = sentences(r.get("reasoning") or "")
        ok_strict = bool(ss); ok_lenient = bool(ss)
        for s in ss:
            total_sent += 1
            nums = [int(x) for g in CIT.findall(s) for x in re.split(r"\s*[,;]\s*", g) if x]
            if nums: cited_sent += 1
            if any(not (1 <= x <= ndocs) for x in nums): bad_num += 1
            ends = bool(re.search(r"\[\d+(?:\s*[,;]\s*\d+)*\]\s*[.!?]?\s*$", s))
            if not (ends and nums and all(1 <= x <= ndocs for x in nums)): ok_strict = False
            if not (nums and all(1 <= x <= ndocs for x in nums)): ok_lenient = False
        strict += ok_strict; lenient += ok_lenient
    n = len(rows)
    print(f"  {nm:14s} every sentence ends with a valid [n]: {strict}/{n} = {100*strict/n:.0f}%  "
          f"(invalid {100*(n-strict)/n:.0f}%)")
    print(f"  {'':14s} every sentence contains a valid [n] anywhere: {lenient}/{n} = {100*lenient/n:.0f}%  "
          f"(invalid {100*(n-lenient)/n:.0f}%)")
    print(f"  {'':14s} sentences {total_sent}, with a citation {cited_sent} ({100*cited_sent/total_sent:.0f}%), "
          f"out-of-range doc numbers {bad_num}")
