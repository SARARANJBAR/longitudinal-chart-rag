"""Stage 07 — run the ablation matrix and build the comparison table.

Core:
  dense_k5          dense only, k=5 (baseline)
  hybrid_k5         + BM25
  date_filtered     metadata pre-filter, then dense

Optional (time / budget permitting):
  fixed512          re-run the Processing Job with CHUNK_STRATEGY=fixed512
  rerank            deploy a cross-encoder to a real-time endpoint, rerank top-20->5, delete

Reads artifacts/results_*.json, writes artifacts/ablation_table.md.
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
