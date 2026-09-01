"""Stage 02 — derive gold truth from Synthea FHIR bundles.

For each (patient, measurement_year) in the CMS122 denominator:
  - answer: most recent HbA1c in the period <= 9%  ->  "controlled" / "not controlled"
  - gold_chunks: encounter id(s) containing the qualifying HbA1c Observation

Output:
  data/eval_set.jsonl   {query_id, patient_id, year, question, answer, gold_chunk_ids}

Risk: FHIR traversal can overrun. Cap at 4h; fallback = hardcode 5 patients.
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
