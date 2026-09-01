"""Stage 02 — derive gold truth from the Synthea CSV export.

For each (patient, measurement_year) where the patient is in the CMS122
denominator (diabetes + a visit that year + age 18-75):
  - answer:      most recent HbA1c (LOINC 4548-4) in that year <= 9%
                 -> "controlled" / "not controlled"
  - gold_chunks: the encounter_id(s) of that qualifying HbA1c observation
                 (observations.csv ENCOUNTER column)

Output:
  data/eval_set.jsonl
    {query_id, patient_id, year, question, answer, gold_chunk_ids, hba1c_value}
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
