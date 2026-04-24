####################################################################################################
###     Imports
####################################################################################################
import argparse
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field

import pickle
import numpy as np
import bm25s
import faiss

from sentence_transformers import SentenceTransformer

import os
from pathlib import Path

from google import genai
from google.genai import types
import json

import re


####################################################################################################
###     Argument Parsing
####################################################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Company search and ranking system")

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User search query",
    )

    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=10,
        help="Number of candidates retrieved independently by BM25 and semantic search before filtering and ranking",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="LLM API key. If omitted, the script will try the GEMINI_API_KEY environment variable.",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save the final ranked results as CSV",
    )

    return parser.parse_args()


####################################################################################################
###     Artifact Loading
####################################################################################################
def load_embedding_model(metadata):
    embedding_model_name = metadata["embedding_model_name"]
    local_model_path = Path("artifacts/embedding_model")

    if local_model_path.exists():
        embedding_model = SentenceTransformer(str(local_model_path))
    else:
        print(f"Downloading embedding model: {embedding_model_name}")
        embedding_model = SentenceTransformer(embedding_model_name)
        local_model_path.mkdir(parents=True, exist_ok=True)
        embedding_model.save(str(local_model_path))
        print(f"Saved embedding model to: {local_model_path}")

    return embedding_model

def load_artifacts():
    companies_df = pd.read_pickle("artifacts/companies_processed.pkl")

    with open("artifacts/bm25_index.pkl", "rb") as f:
        bm25_index = pickle.load(f)

    faiss_index = faiss.read_index("artifacts/company_faiss.index")

    with open("artifacts/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    embedding_model = load_embedding_model(metadata)

    return {
        "companies_df": companies_df,
        "bm25_index": bm25_index,
        "faiss_index": faiss_index,
        "metadata": metadata,
        "embedding_model": embedding_model,
    }


####################################################################################################
###     Schemas
####################################################################################################
class HardFilters(BaseModel):
    country_codes: List[str] = Field(default_factory=list)
    year_founded_min: Optional[int] = None
    year_founded_max: Optional[int] = None
    revenue_min: Optional[float] = None
    revenue_max: Optional[float] = None
    employee_count_min: Optional[int] = None
    employee_count_max: Optional[int] = None
    is_public: Optional[bool] = None

class QueryPlan(BaseModel):
    hard_filters: HardFilters
    countries: List[str] = Field(default_factory=list)
    regions: List[str] = Field(default_factory=list)
    industry_terms: List[str] = Field(default_factory=list)
    naics_terms: List[str] = Field(default_factory=list)
    business_model_terms: List[str] = Field(default_factory=list)
    offering_terms: List[str] = Field(default_factory=list)
    target_market_terms: List[str] = Field(default_factory=list)
    canonical_terms: List[str] = Field(default_factory=list)
    exclude_terms: List[str] = Field(default_factory=list)
    target_description: str


####################################################################################################
###     Query Parsing
####################################################################################################
def resolve_api_key(args):
    if args.api_key:
        return args.api_key

    env_api_key = os.getenv("GEMINI_API_KEY")
    if env_api_key:
        return env_api_key

    raise ValueError(
        "Missing API key. Provide --api-key or set GEMINI_API_KEY in the environment."
    )


SYSTEM_PROMPT = """
You are an expert query parser for a company search engine.

Your task is to convert a user query into a structured search plan.

The company database may contain fields such as:
- year_founded
- address
- employee_count
- revenue
- is_public
- primary_naics
- secondary_naics
- description
- business_model
- target_markets
- core_offerings

Rules:
- Extract only constraints explicitly stated or strongly implied by the query.
- Do not invent unsupported facts, thresholds, geographies, or attributes.
- For numeric constraints, use min/max style fields when appropriate.
- Generate business-language terms likely to appear in description, NAICS labels, business_model, target_markets, and core_offerings.
- Generate offering-related terms when the query implies products or services.
- Generate target-market terms when the query implies customer segments or served industries.
- Use exclude_terms only when the query clearly implies exclusions.
- Generate a short target_description suitable for semantic retrieval.
- Preserve ambiguity when the query is vague.

- Extract geographic mentions into top-level QueryPlan geography fields:
  - put explicit or strongly implied countries into countries
  - put explicit geographic regions, continents, economic blocs, or political groupings into regions

- hard_filters.country_codes is the executable geography filter for search-time filtering.

- When the query specifies a country:
  - add the country name to countries
  - add its ISO 3166-1 alpha-2 code to hard_filters.country_codes

- When the query specifies a city or locality:
  - infer its country only if the mapping is clear and unambiguous in context
  - if so, add the country name to countries
  - and add its ISO 3166-1 alpha-2 code to hard_filters.country_codes
  - otherwise, do not invent a country code

- When the query specifies a geographic region, continent, economic bloc, or political grouping
  (for example Europe, Scandinavia, Nordics, EU, NATO, BRICS, DACH):
  - preserve the original mention in regions
  - expand it into the corresponding ISO 3166-1 alpha-2 country codes
  - place those codes in hard_filters.country_codes

- Only include geographic filters when the geography is explicit or strongly implied.
- Do not invent unsupported geographic constraints.
- If a geographic mention is ambiguous, preserve the ambiguity and only add country codes when the mapping is reliable.
- hard_filters.country_codes must contain unique uppercase ISO 3166-1 alpha-2 codes.

- Return valid JSON only.
"""


def parse_query(user_query, api_key, model_name="gemini-2.5-flash"):
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_name,
        contents=user_query,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=QueryPlan,
            temperature=0.0,
        ),
    )

    return response.parsed


####################################################################################################
###     Query Text Builders
####################################################################################################
def deduplicate_preserve_order(values):
    unique_values = []
    seen_values = set()

    for value in values:
        if value is None:
            continue
        value = str(value).strip()
        if not value:
            continue
        normalized_value = value.lower()
        if normalized_value in seen_values:
            continue
        seen_values.add(normalized_value)
        unique_values.append(value)

    return unique_values


def build_bm25_query(query_plan: QueryPlan) -> str:
    bm25_terms = []
    bm25_terms.extend(query_plan.countries)
    bm25_terms.extend(query_plan.regions)
    bm25_terms.extend(query_plan.industry_terms)
    bm25_terms.extend(query_plan.naics_terms)
    bm25_terms.extend(query_plan.business_model_terms)
    bm25_terms.extend(query_plan.offering_terms)
    bm25_terms.extend(query_plan.target_market_terms)
    bm25_terms.extend(query_plan.canonical_terms)

    bm25_terms = deduplicate_preserve_order(bm25_terms)

    return " ".join(bm25_terms)


def build_semantic_query(query_plan: QueryPlan) -> str:
    return query_plan.target_description.strip()


def build_query_texts(query_plan: QueryPlan):
    return {
        "bm25_query": build_bm25_query(query_plan),
        "semantic_query": build_semantic_query(query_plan),
    }


####################################################################################################
###     Retrieval
####################################################################################################
def retrieve_bm25_candidates(bm25_query, artifacts, top_k):
    bm25_index = artifacts["bm25_index"]

    query_tokens = bm25s.tokenize([bm25_query])
    bm25_results, bm25_scores = bm25_index.retrieve(query_tokens, k=top_k)

    company_ids = bm25_results[0]
    scores = bm25_scores[0]
    bm25_table = pd.DataFrame(
        {
            "company_id": company_ids,
            "bm25_score": scores,
        }
    )

    return bm25_table


def retrieve_semantic_candidates(semantic_query, artifacts, top_k):
    faiss_index = artifacts["faiss_index"]
    embedding_model = artifacts["embedding_model"]

    query_embedding = embedding_model.encode(
        [semantic_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    semantic_scores, semantic_ids = faiss_index.search(query_embedding, top_k)

    company_ids = semantic_ids[0]
    scores = semantic_scores[0]
    semantic_table = pd.DataFrame(
        {
            "company_id": company_ids,
            "semantic_score": scores,
        }
    )

    return semantic_table


def min_max_normalize(series):
    min_value = series.min()
    max_value = series.max()

    if pd.isna(min_value) or pd.isna(max_value):
        return pd.Series([0.0] * len(series), index=series.index)

    if max_value - min_value < 1e-12:
        return pd.Series([1.0] * len(series), index=series.index)

    return (series - min_value) / (max_value - min_value)


def fuse_candidate_tables(bm25_table, semantic_table):
    candidate_table = bm25_table.merge(
        semantic_table,
        on="company_id",
        how="outer",
    )

    candidate_table["bm25_score"] = candidate_table["bm25_score"].fillna(0.0)
    candidate_table["semantic_score"] = candidate_table["semantic_score"].fillna(0.0)

    candidate_table["bm25_score_norm"] = min_max_normalize(candidate_table["bm25_score"])
    candidate_table["semantic_score_norm"] = min_max_normalize(candidate_table["semantic_score"])

    candidate_table["retrieval_score"] = (
        0.5 * candidate_table["bm25_score_norm"] +
        0.5 * candidate_table["semantic_score_norm"]
    )

    candidate_table = candidate_table.sort_values(
        "retrieval_score",
        ascending=False,
    ).reset_index(drop=True)

    return candidate_table


def retrieve_candidates(query_texts, artifacts, top_k):
    bm25_table = retrieve_bm25_candidates(
        bm25_query=query_texts["bm25_query"],
        artifacts=artifacts,
        top_k=top_k,
    )

    semantic_table = retrieve_semantic_candidates(
        semantic_query=query_texts["semantic_query"],
        artifacts=artifacts,
        top_k=top_k,
    )

    candidate_table = fuse_candidate_tables(
        bm25_table=bm25_table,
        semantic_table=semantic_table,
    )

    return candidate_table


def attach_company_columns(candidate_table, artifacts):
    companies_df = artifacts["companies_df"]

    columns_to_attach = [
        "website",
        "operational_name",
        "year_founded",
        "address",
        "employee_count",
        "revenue",
        "is_public",
        "primary_naics",
        "secondary_naics",
        "description",
        "business_model",
        "target_markets",
        "core_offerings",
    ]

    candidate_table = candidate_table.copy()

    for column_name in columns_to_attach:
        candidate_table[column_name] = candidate_table["company_id"].map(
            companies_df[column_name]
        )

    return candidate_table


####################################################################################################
###     Hard Filters
####################################################################################################
def is_missing(value):
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return False


def passes_bool_filter(value, required_value):
    if required_value is None:
        return True
    if is_missing(value):
        return False
    return bool(value) == required_value


def passes_min_filter(value, min_value):
    if min_value is None:
        return True
    if is_missing(value):
        return False
    return value >= min_value


def passes_max_filter(value, max_value):
    if max_value is None:
        return True
    if is_missing(value):
        return False
    return value <= max_value


def passes_range_filter(value, min_value=None, max_value=None):
    return passes_min_filter(value, min_value) and passes_max_filter(value, max_value)


def normalize_country_code(country_code):
    if is_missing(country_code):
        return None
    country_code = str(country_code).strip().upper()
    if not country_code:
        return None
    return country_code


def apply_hard_filters_to_row(row, hard_filters):
    failed_reasons = []

    company_country_code = None
    address = row["address"]
    if isinstance(address, dict):
        company_country_code = normalize_country_code(address.get("country_code"))

    allowed_country_codes = [
        normalize_country_code(code)
        for code in hard_filters.country_codes
    ]
    allowed_country_codes = [code for code in allowed_country_codes if code is not None]

    if allowed_country_codes:
        if company_country_code is None:
            failed_reasons.append("country_code_missing")
        elif company_country_code not in allowed_country_codes:
            failed_reasons.append("country_code_not_allowed")

    if not passes_bool_filter(row["is_public"], hard_filters.is_public):
        failed_reasons.append("is_public_mismatch_or_missing")

    if not passes_range_filter(
        row["year_founded"],
        hard_filters.year_founded_min,
        hard_filters.year_founded_max,
    ):
        failed_reasons.append("year_founded_out_of_range_or_missing")

    if not passes_range_filter(
        row["revenue"],
        hard_filters.revenue_min,
        hard_filters.revenue_max,
    ):
        failed_reasons.append("revenue_out_of_range_or_missing")

    if not passes_range_filter(
        row["employee_count"],
        hard_filters.employee_count_min,
        hard_filters.employee_count_max,
    ):
        failed_reasons.append("employee_count_out_of_range_or_missing")

    return {
        "company_country_code": company_country_code,
        "passed_hard_filters": len(failed_reasons) == 0,
        "hard_filter_failed_reasons": failed_reasons,
    }


def apply_hard_filters(candidate_table, query_plan):
    filter_results = candidate_table.apply(
        lambda row: apply_hard_filters_to_row(row, query_plan.hard_filters),
        axis=1,
    )

    filter_results_df = pd.DataFrame(filter_results.tolist())

    candidate_table = candidate_table.reset_index(drop=True)
    filter_results_df = filter_results_df.reset_index(drop=True)

    candidate_table["company_country_code"] = filter_results_df["company_country_code"]
    candidate_table["passed_hard_filters"] = filter_results_df["passed_hard_filters"]
    candidate_table["hard_filter_failed_reasons"] = filter_results_df["hard_filter_failed_reasons"]

    return candidate_table


def keep_only_passing_candidates(candidate_table):
    candidate_table = candidate_table[candidate_table["passed_hard_filters"]].copy()
    candidate_table = candidate_table.sort_values(
        "retrieval_score",
        ascending=False,
    ).reset_index(drop=True)

    return candidate_table


####################################################################################################
###     Scoring
####################################################################################################
def normalize_text(text):
    if text is None:
        return ""
    try:
        if pd.isna(text):
            return ""
    except Exception:
        pass
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)

    return text


def list_field_to_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return normalize_text(" ".join(str(item) for item in value if item is not None))

    return normalize_text(str(value))


def naics_entry_to_text(value):
    if value is None:
        return ""
    if isinstance(value, dict):
        parts = []
        label = value.get("label")
        code = value.get("code")
        if label is not None:
            parts.append(str(label))
        if code is not None:
            parts.append(str(code))
        return normalize_text(" ".join(parts))

    return normalize_text(str(value))


def get_description_text(row):
    return normalize_text(row["description"])


def get_naics_text(row):
    primary_text = naics_entry_to_text(row["primary_naics"])
    secondary_text = naics_entry_to_text(row["secondary_naics"])

    return normalize_text(f"{primary_text} {secondary_text}")


def get_business_model_text(row):
    return list_field_to_text(row["business_model"])


def get_offerings_text(row):
    return list_field_to_text(row["core_offerings"])


def get_target_markets_text(row):
    return list_field_to_text(row["target_markets"])


def deduplicate_terms(terms):
    unique_terms = []
    seen_terms = set()

    for term in terms:
        if term is None:
            continue
        term = normalize_text(term)
        if not term:
            continue
        if term in seen_terms:
            continue
        seen_terms.add(term)
        unique_terms.append(term)

    return unique_terms

def compute_term_match_score(terms, text):
    text = normalize_text(text)
    clean_terms = deduplicate_terms(terms)

    if not clean_terms:
        return 0.0

    matched_terms = 0
    for term in clean_terms:
        if term in text:
            matched_terms += 1

    return matched_terms / len(clean_terms)


def compute_industry_score(row, query_plan):
    industry_terms = (
        query_plan.industry_terms +
        query_plan.naics_terms +
        query_plan.canonical_terms
    )

    naics_text = get_naics_text(row)
    description_text = get_description_text(row)

    naics_score = compute_term_match_score(industry_terms, naics_text)
    description_score = compute_term_match_score(industry_terms, description_text)

    return 0.6 * naics_score + 0.4 * description_score


def compute_business_model_score(row, query_plan):
    business_model_text = get_business_model_text(row)
    description_text = get_description_text(row)

    business_model_score = compute_term_match_score(
        query_plan.business_model_terms,
        business_model_text,
    )

    description_score = compute_term_match_score(
        query_plan.business_model_terms,
        description_text,
    )

    return 0.7 * business_model_score + 0.3 * description_score


def compute_offering_score(row, query_plan):
    offerings_text = get_offerings_text(row)
    description_text = get_description_text(row)

    offerings_score = compute_term_match_score(
        query_plan.offering_terms,
        offerings_text,
    )

    description_score = compute_term_match_score(
        query_plan.offering_terms,
        description_text,
    )

    return 0.7 * offerings_score + 0.3 * description_score


def compute_target_market_score(row, query_plan):
    target_markets_text = get_target_markets_text(row)
    description_text = get_description_text(row)

    target_markets_score = compute_term_match_score(
        query_plan.target_market_terms,
        target_markets_text,
    )

    description_score = compute_term_match_score(
        query_plan.target_market_terms,
        description_text,
    )

    return 0.7 * target_markets_score + 0.3 * description_score


def compute_exclude_penalty(row, query_plan):
    full_text = " ".join(
        [
            get_description_text(row),
            get_naics_text(row),
            get_business_model_text(row),
            get_offerings_text(row),
            get_target_markets_text(row),
        ]
    )

    return compute_term_match_score(query_plan.exclude_terms, full_text)


def score_candidate_row(row, query_plan):
    retrieval_score = row["retrieval_score"]

    industry_score = compute_industry_score(row, query_plan)
    business_model_score = compute_business_model_score(row, query_plan)
    offering_score = compute_offering_score(row, query_plan)
    target_market_score = compute_target_market_score(row, query_plan)
    exclude_penalty = compute_exclude_penalty(row, query_plan)

    final_score = (
        0.40 * retrieval_score +
        0.20 * industry_score +
        0.15 * business_model_score +
        0.15 * offering_score +
        0.10 * target_market_score -
        0.20 * exclude_penalty
    )

    return {
        "industry_score": industry_score,
        "business_model_score": business_model_score,
        "offering_score": offering_score,
        "target_market_score": target_market_score,
        "exclude_penalty": exclude_penalty,
        "final_score": final_score,
    }


def score_candidates(candidate_table, query_plan):
    score_results = candidate_table.apply(
        lambda row: score_candidate_row(row, query_plan),
        axis=1,
    )

    score_results_df = pd.DataFrame(score_results.tolist())
    candidate_table = candidate_table.reset_index(drop=True)
    score_results_df = score_results_df.reset_index(drop=True)
    candidate_table["industry_score"] = score_results_df["industry_score"]
    candidate_table["business_model_score"] = score_results_df["business_model_score"]
    candidate_table["offering_score"] = score_results_df["offering_score"]
    candidate_table["target_market_score"] = score_results_df["target_market_score"]
    candidate_table["exclude_penalty"] = score_results_df["exclude_penalty"]
    candidate_table["final_score"] = score_results_df["final_score"]

    return candidate_table


def rank_candidates(candidate_table):
    candidate_table = candidate_table.sort_values(
        "final_score",
        ascending=False,
    ).reset_index(drop=True)

    return candidate_table


####################################################################################################
###     Output
####################################################################################################
def build_output_table(candidate_table):
    output_columns = [
        "operational_name",
        "website",
        "company_country_code",
        "final_score",
        "retrieval_score",
        "year_founded",
        "employee_count",
        "revenue",
        "is_public",
        "description",
    ]

    existing_columns = [
        column_name
        for column_name in output_columns
        if column_name in candidate_table.columns
    ]

    return candidate_table[existing_columns].copy()


def print_final_results(output_table):
    print("\nFINAL RESULTS:")
    print(output_table.to_string())


def export_results_to_csv(output_table, output_csv_path):
    if output_csv_path is None:
        return

    output_table.to_csv(output_csv_path, index=False)
    print(f"\nSaved results to: {output_csv_path}")


####################################################################################################
###     Main
####################################################################################################
def main():
    # Setup
    args = parse_args()
    user_query = args.query
    top_k = args.retrieval_k
    output_csv_path = args.output_csv
    api_key = resolve_api_key(args)
    artifacts = load_artifacts()

    # Step 1 - Query Parsing
    query_plan = parse_query(user_query, api_key)

    # Step 2 - Hybrid Candidate Retrieval
    query_texts = build_query_texts(query_plan)
    candidate_table = retrieve_candidates(
        query_texts=query_texts,
        artifacts=artifacts,
        top_k=top_k,
    )
    candidate_table = attach_company_columns(
        candidate_table=candidate_table,
        artifacts=artifacts,
    )

    # Step 3 - Hard Constraint Filtering
    candidate_table = apply_hard_filters(
        candidate_table=candidate_table,
        query_plan=query_plan,
    )
    candidate_table = keep_only_passing_candidates(candidate_table)

    # Step 4 - Feature-Based Candidate Scoring
    candidate_table = score_candidates(
        candidate_table=candidate_table,
        query_plan=query_plan,
    )
    candidate_table = rank_candidates(candidate_table)

    # Output
    output_table = build_output_table(candidate_table)
    print_final_results(output_table)
    export_results_to_csv(output_table, output_csv_path)


if __name__ == "__main__":
    main()
