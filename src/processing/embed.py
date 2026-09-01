"""Runs INSIDE the SageMaker Processing Job container.

Reads   /opt/ml/processing/input/   (corpus notes + eval_set.jsonl)
Chunks  per encounter (CHUNK_STRATEGY; "fixed512" variant for the ablation)
Embeds  with BAAI/bge-small-en-v1.5 (CPU)
Writes  /opt/ml/processing/output/
          index.faiss              IndexFlatIP over chunk embeddings
          chunks.parquet           chunk_id, patient_id, encounter_date, type, text
          query_embeddings.parquet query_id, embedding

SageMaker maps the output dir to S3 automatically.
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
