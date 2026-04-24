# Veridion Intent Qualification Challenge

This repository contains my solution for the **Veridion Intent Qualification Challenge**.

The goal is to build a ranking and qualification system that determines which companies truly match a user query, while balancing **accuracy, speed, cost, and scalability**.

---

## Repository Structure

- `solution.py` — main runtime pipeline used to process a user query and return ranked candidate companies
- `prepare_index.ipynb` — offline preprocessing notebook used to:
  - clean and flatten company records
  - build the company retrieval text
  - generate retrieval artifacts
  - create the BM25 and FAISS search indexes
- `requirements.txt` — Python dependencies
- `WRITEUP.md` — detailed explanation of the approach, tradeoffs, limitations, and future improvements
- `artifacts/` — precomputed data and retrieval indexes used by `solution.py`

---

## Pipeline Overview

The system follows a multi-stage pipeline:

![Pipeline Diagram](images/pipeline_diagram.png)

1. **Query Parsing**  
   The raw user query is parsed with an LLM into a structured `QueryPlan`.

2. **Hybrid Candidate Retrieval**  
   A candidate shortlist is retrieved using:
   - **BM25** for lexical matching
   - **FAISS + sentence embeddings** for semantic matching

3. **Hard Constraint Filtering**  
   Structured constraints such as geography, company size, revenue, founding year, and public/private status are applied to the retrieved candidates.

4. **Feature-Based Candidate Scoring**  
   Remaining candidates are scored using interpretable relevance features such as:
   - industry match
   - business model match
   - offering match
   - target market match
   - exclusion penalty

---

## Setup & Usage

Install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The solution is executed through `solution.py`:

```bash
python solution.py \
  --query "Find B2B SaaS companies in Germany selling cybersecurity solutions"
```

If the Gemini API key is not passed directly, the script reads it from the `GEMINI_API_KEY` environment variable:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

You can also provide it explicitly:

```bash
python solution.py \
  --query "Find private fintech companies in France" \
  --api-key "your_api_key_here"
```

### Arguments

- `--query` — required; natural-language company search query.
- `--retrieval-k` — optional; number of candidates retrieved by BM25 and semantic search before filtering. Default: `10`.
- `--api-key` — optional; Gemini API key. If omitted, `GEMINI_API_KEY` is used.
- `--output-csv` — optional; path where the final ranked results are saved as CSV.

Example with CSV export:

```bash
python solution.py \
  --query "Find logistics technology companies in Europe" \
  --retrieval-k 25 \
  --output-csv outputs/results.csv
```

The script prints the final ranked companies in the terminal and, if `--output-csv` is provided, also saves them to a CSV file.

The output includes company name, website, country code, final score, retrieval score, founding year, employee count, revenue, public/private status, and description when available.