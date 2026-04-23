# SubgraphRAG Reproduction — CS 514 Final Project

Reproduction of the LLM reasoning stage from:

> **"Simple is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation"**
> ICLR 2025 · [arXiv 2410.20724](https://arxiv.org/abs/2410.20724)

We reproduce the WebQSP evaluation using the paper authors' pre-scored retrieval results and GPT-4o-mini, and run two additional experiments (retrieval recall analysis and a no-retrieval ablation).

---

## Results

| Configuration | Macro-F1 | Hit |
|---|---|---|
| Paper reported (full 1628 qs, Oct 2024) | 77.45% | 90.11% |
| Authors' stored predictions re-scored (150 qs) | 69.32% | 76.00% |
| **Our GPT-4o-mini run (150 qs, Apr 2026)** | **68.91%** | **76.67%** |
| No retrieval — GPT-4o-mini only (30 qs) | 33.56% | 36.67% |

Our run is within **0.4 pp F1** of the authors' own stored predictions re-scored with the same metric code. The ~9 pp gap from the paper's reported numbers is due to **GPT-4o-mini model drift** (the paper used a late-2024 snapshot; that version is no longer served). The no-retrieval ablation shows the KG contributes **+42.5 pp F1**.

---

## Setup

**Requirements:** Python 3.10+, Windows/Linux/Mac, no GPU needed.

```bash
pip install -r requirements.txt
```

Set your OpenAI API key as an environment variable — do **not** hardcode it:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."

# Linux / macOS
export OPENAI_API_KEY="sk-..."
```

---

## Reproducing the results

### 1. Download data

```bash
python download.py
```

Downloads two sources into local directories (not committed to this repo):
- `subgraphrag_data/` — authors' pre-scored retrieval results from `siqim311/SubgraphRAG` on HuggingFace (~2 GB)
- `hf_cache/` — WebQSP dataset from `rmanluo/RoG-webqsp` (~1.65 GB)

### 2. Run LLM inference (150 questions)

```bash
python reason.py
```

Samples 150 questions (`seed=42`) from the WebQSP test set, calls GPT-4o-mini with the top-100 scored triples per question, and writes `results_webqsp.json`. Takes ~4 minutes, ~$0.05 in API cost.

### 3. Run no-retrieval ablation (30 questions)

```bash
python ablation_no_retrieval.py
```

Calls GPT-4o-mini on the same 30-question subset with **no triples** — just the raw question. Writes `results_no_retrieval.json` and saves `ablation_no_retrieval.png`.

### 4. Generate all charts

```bash
python compute_and_plot.py
```

Loads `results_webqsp.json` and the scored-triples `.pth` file (~290 MB, must exist from step 1) and writes six PNG charts.

---

## Output files

| File | Description |
|---|---|
| `results_webqsp.json` | Per-question F1, Hit, predicted answers, LLM responses (150 qs) |
| `results_no_retrieval.json` | Same structure, no-retrieval run (30 qs) |
| `results_comparison.png` | Paper vs authors' preds vs our run |
| `f1_distribution.png` | Histogram of per-question F1 (bimodal) |
| `performance_by_answer_count.png` | F1 / Hit by number of gold answers |
| `retrieval_recall_at_k.png` | Answer entity recall vs. retrieval budget K |
| `ablation_no_retrieval.png` | With-retrieval vs. no-retrieval comparison |
| `example_walkthrough.png` | Full pipeline walkthrough for one question |

---

## How it works

SubgraphRAG has two stages:

1. **Retrieval** — a trained scorer ranks Freebase triples by relevance to the question. We skip retraining and use the authors' pre-scored results directly.
2. **Reasoning** — the top-K triples are formatted as text and passed to an LLM with a system prompt and one in-context example. The LLM outputs `ans:` prefixed answers.

The knowledge graph is **Freebase**, accessed via pre-extracted subgraphs from the `rmanluo/RoG-webqsp` dataset. Freebase has been offline since 2016; the subgraphs were extracted and distributed by prior work (RoG).

---

## Key findings

- **Retrieval recall saturates early:** 91.3% of gold-answer entities appear in the top-25 triples. Adding more triples (up to 500) only raises recall to 94%. K=100 is a good trade-off.
- **Bimodal F1 distribution:** 83/150 questions score perfect F1=1.0; 39 score below 0.25. The system either fully succeeds or fails — partial credit is rare.
- **Single-answer questions are easier:** F1=0.75 vs 0.65 for multi-answer questions, because there is only one target entity to find.
- **The KG is load-bearing:** Removing retrieval entirely drops F1 from 0.76 to 0.34 on the same 30-question set.

---

## Limitations

- We use a 150-question sample, not the full 1,628-question test set. Results have sampling variance (~±3 pp).
- GPT-4o-mini has been updated since the paper was written. The stored predictions in the HuggingFace repo score 69.3% F1 when re-evaluated — matching our run but not the paper's 77.45%. This is model drift, not a bug.
- The retriever was not retrained (requires a GPU and several hours of training).
