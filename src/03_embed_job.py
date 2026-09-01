"""Stage 03 — launch the transient SageMaker Processing Job.

Uploads data/corpus/ + data/eval_set.jsonl to s3://$BUCKET/$PREFIX/input/,
runs src/processing/embed.py on a prebuilt PyTorch container (ml.m5.xlarge, CPU)
with sentence-transformers + faiss-cpu installed from requirements, then the job
writes index.faiss + chunks.parquet + query_embeddings.parquet to
s3://$BUCKET/$PREFIX/output/ and terminates.

This is the core SageMaker artifact: batch, containerized, S3 in/out, no idle cost.
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
