# MedTrustEval evaluation code

These scripts score the LLM-only and LLM + RAG runs on all seven MedTrustEval dimensions. Put this folder's files in `~/Projects/MedTrustEval/` next to the `logs/` folder.

## 1. Setup (once)

```bash
cd ~/Projects/MedTrustEval
pip install torch transformers matplotlib   # only needed for hallucination/grounding (NLI) and the diagram
python -m unittest discover -p "test_*.py"  # 26 tests should pass
```

The first NLI run downloads the free model `cross-encoder/nli-deberta-v3-base` (about 0.7 GB) from Hugging Face.

## 2. Run everything

```bash
python run_all.py --llm logs/llm_only_results.jsonl --rag logs/rag_results.jsonl --out results
```

This writes `results/RESULTS_TABLES.md`, which holds Tables 3–5 for the paper, plus the detailed files listed below.
If PyTorch isn't installed yet, add `--skip-hallucination`.

## 3. Scripts

| Script | Dimension(s) | Needs |
|---|---|---|
| `evaluate_accuracy.py` | 1 Accuracy: CI, per dataset, letter bias, RAG helped/hurt, McNemar | the two logs |
| `evaluate_hallucination_grounding.py` | 2 Hallucination, 3 Grounding, 4 Retrieval (evidence hit), 5 Citations | the two logs, torch + transformers |
| `evaluate_calibration.py` | 7 Calibration: ECE, Brier, confidence gap, reliability diagram | the two logs |
| `evaluate_consistency.py` | 6 Consistency: agreement across repeat runs, shuffle stability | repeat runs and a shuffled run (see §4) |
| `make_shuffled_questions.py` | makes the shuffled-options question file for the consistency test | question file |
| `review_sheets.py` | manual review: example sheets, Precision@5, Cohen's kappa, NLI vs. human check | results from the scripts above |
| `make_review_workbook.py` | one Excel workbook for the Day 9 review (examples, retrieval relevance, statement labels, auto-scoring) | the two logs only |
| `run_experiments.py` | runs llama3:8b via Ollama yourself (new prompt, repeat runs, shuffled options), reusing the passages saved in `rag_results.jsonl` | Ollama + `ollama pull llama3:8b` |
| `run_all.py` | runs everything and writes the result tables | |
| `mte_common.py`, `nli_judge.py` | shared helpers (loading, sentence splitting, citations, NLI model) | |
| `make_mock_logs.py`, `test_*.py` | fake data and tests | |

## 4. What Justin's pipeline needs to produce

| For | Files | Notes |
|---|---|---|
| Dimensions 1–5 and 7 | `logs/llm_only_results.jsonl`, `logs/rag_results.jsonl` | Already produced |
| Consistency (repeat) | `logs/repeats/llm_only_run1.jsonl` … `run5.jsonl`, and the same for `rag_run*` | Same questions and settings, run 5 times (run 1 can be a copy of the main log) |
| Consistency (shuffle) | `logs/shuffled/llm_only_shuffled.jsonl`, `logs/shuffled/rag_shuffled.jsonl` | Run on the file from `make_shuffled_questions.py`. **Keep the `option_map` field in each output row.** |

Every row must keep `dataset` and the exact `question` text. The scripts use those two fields to match the same question across files.

## 5. Manual review (Day 9)

```bash
python review_sheets.py make --out review      # creates CSVs to fill in, in Excel or Google Sheets
# both raters fill in the rater1_/rater2_ columns
python review_sheets.py score --out review     # Precision@5, kappa, and NLI-vs-human agreement
```

Don't open `_statement_auto_labels_DO_NOT_OPEN_BEFORE_RATING.csv` until both of you have finished labelling.

## 6. Scoring rules

- **Answers:** both answers are converted to an option letter. Answers that can't be read count as wrong and are reported separately.
- **Statements:** each explanation sentence (with citation markers removed) is checked against the 5 retrieved passages.
  - *Supported:* entailment ≥ 0.5.
  - *Contradicted:* not supported, and contradiction ≥ 0.5.
  - *Not addressed:* anything else. This is **not** counted as a hallucination.
- **LLM-only answers:** these are checked against the same passages that were retrieved for RAG.
- **Confident error:** a wrong answer given with confidence ≥ 80.

`--judge lexical` runs the pipeline without PyTorch, but its numbers are meaningless, so use it only for testing.
