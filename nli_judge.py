"""
Judges that decide whether a piece of evidence supports a statement.

NLIJudge      – the real judge. A free, open-source natural-language-inference model
                (default: cross-encoder/nli-deberta-v3-base) run locally with
                Hugging Face transformers. Needs:  pip install torch transformers
LexicalJudge  – a crude word-overlap stand-in with NO medical understanding.
                Only for testing the pipeline when transformers isn't installed.
                Never report results from it.

Both return, for each (premise, hypothesis) pair, a dict:
    {"entailment": p, "contradiction": p, "neutral": p}
"""
from __future__ import annotations

import re

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"


class NLIJudge:
    name = "nli"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 batch_size: int = 16, max_length: int = 512):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                "The NLI judge needs PyTorch and transformers:\n"
                "    pip install torch transformers\n"
                "(or run with --judge lexical just to test the pipeline)"
            ) from e
        self.torch = torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_name = model_name
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.idx = {}
        for i, lab in id2label.items():
            for key in ("entailment", "contradiction", "neutral"):
                if lab.startswith(key[:5]):
                    self.idx[key] = i
        if set(self.idx) != {"entailment", "contradiction", "neutral"}:
            raise SystemExit(f"Unexpected labels for {model_name}: {id2label}")
        self._cache: dict[tuple[str, str], dict] = {}

    def score(self, pairs: list[tuple[str, str]]) -> list[dict]:
        todo = [p for p in dict.fromkeys(pairs) if p not in self._cache]
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i:i + self.batch_size]
            enc = self.tok([p for p, _ in batch], [h for _, h in batch], truncation="only_first",
                           max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probs = self.torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().tolist()
            for pair, pr in zip(batch, probs):
                self._cache[pair] = {k: pr[j] for k, j in self.idx.items()}
        return [self._cache[p] for p in pairs]


_STOP = set("""a an the of and or to in on for with by is are was were be been it this that these those as at from
which who whom whose what when where why how not no than then there their its his her they them he she we you i
can may might will would should could has have had do does did most more less very also such into""".split())


def _toks(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 2}


class LexicalJudge:
    """TEST ONLY. Word overlap; a negation mismatch is treated as contradiction."""
    name = "lexical"
    model_name = "lexical-overlap (test only)"

    def score(self, pairs):
        out = []
        for prem, hyp in pairs:
            h, p = _toks(hyp), _toks(prem)
            overlap = len(h & p) / len(h) if h else 0.0
            neg_h = bool(re.search(r"\b(not|no|never|without)\b", hyp.lower()))
            neg_p = bool(re.search(r"\b(not|no|never|without)\b", prem.lower()))
            if overlap >= 0.6 and neg_h != neg_p:
                out.append({"entailment": 0.05, "contradiction": 0.9, "neutral": 0.05})
            elif overlap >= 0.6:
                out.append({"entailment": 0.9, "contradiction": 0.02, "neutral": 0.08})
            else:
                out.append({"entailment": 0.1, "contradiction": 0.05, "neutral": 0.85})
        return out


def make_judge(kind: str, model_name: str | None = None, device: str | None = None):
    if kind == "nli":
        return NLIJudge(model_name or DEFAULT_MODEL, device=device)
    if kind == "lexical":
        return LexicalJudge()
    raise SystemExit(f"Unknown judge: {kind}")
