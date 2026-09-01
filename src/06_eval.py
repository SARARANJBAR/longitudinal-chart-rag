"""Stage 06 — eval harness.

For a given retrieval mode, over data/eval_set.jsonl:
  retrieval:  recall@{1,3,5}  (gold_chunk_ids vs retrieved)
  end-to-end: answer accuracy (exact controlled/not-controlled match)
  secondary:  citation correctness (cited_chunk_ids overlap gold)

Writes artifacts/results_<mode>.json.
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
