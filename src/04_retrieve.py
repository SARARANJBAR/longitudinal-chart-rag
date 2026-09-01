"""Stage 04 — retrieval.

Pulls index.faiss + chunks.parquet + query_embeddings.parquet from S3 output/.
Provides:
  dense(query_id, k)                 FAISS IndexFlatIP
  hybrid(query_id, k)                dense + rank_bm25, score fusion
  date_filtered(query_id, k, year)   restrict to encounters in the measurement year, then dense

Returns ranked chunk_ids for the eval harness.
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
