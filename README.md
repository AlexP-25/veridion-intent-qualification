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
