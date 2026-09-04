"""Stage 06 — eval harness.

For a given retrieval mode, over data/eval_set.jsonl:
  retrieval:  recall@{1,3,5}  (gold_chunk_ids vs retrieved encounter_ids)
  end-to-end: answer accuracy (hba1c_value within tolerance of gold, e.g. ±0.1%
              — exact definition TBD, see TODO.md Stage 06 decisions)
  secondary:  citation correctness (cited_encounter_ids overlap gold), control_flag match

Writes artifacts/results_<mode>.json.
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
