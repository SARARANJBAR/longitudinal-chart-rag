"""Stage 03 — build the embedding index.

  python src/03_embed_job.py --local     chunk + embed locally -> artifacts/   (approach A: dev loop)
  python src/03_embed_job.py             launch the transient SageMaker Processing Job

The job runs src/processing/embed.py on a prebuilt PyTorch container (ml.t3.medium, CPU),
uploads data/corpus/ + data/eval_set.jsonl as input, writes index.faiss + chunks.parquet
+ query_embeddings.parquet to s3://$RAG_S3_BUCKET/$PREFIX/output/, and terminates.
NOTE: the cloud path is not yet exercised — needs the AWS prerequisites in TODO.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402

PROCESSING_DIR = ROOT / "src" / "processing"


def run_local() -> None:
    from processing.embed import run

    run(ROOT / cfg.DATA_DIR, ROOT / cfg.ARTIFACT_DIR)


def run_job() -> None:
    from sagemaker.processing import ProcessingInput, ProcessingOutput
    from sagemaker.pytorch import PyTorchProcessor

    if not (cfg.S3_BUCKET and cfg.SM_EXECUTION_ROLE):
        sys.exit("Set RAG_S3_BUCKET and SM_EXECUTION_ROLE in .env first (see TODO.md).")

    out_uri = f"s3://{cfg.S3_BUCKET}/{cfg.S3_PREFIX}/output"
    processor = PyTorchProcessor(
        framework_version="2.3",
        py_version="py311",
        role=cfg.SM_EXECUTION_ROLE,
        instance_type=cfg.EMBED_INSTANCE,
        instance_count=1,
        base_job_name="chart-rag-embed",
    )
    processor.run(
        code="embed.py",
        source_dir=str(PROCESSING_DIR),
        inputs=[
            ProcessingInput(source=str(ROOT / cfg.DATA_DIR / "corpus"),
                            destination="/opt/ml/processing/input/corpus"),
            ProcessingInput(source=str(ROOT / cfg.DATA_DIR / "eval_set.jsonl"),
                            destination="/opt/ml/processing/input"),
        ],
        outputs=[ProcessingOutput(source="/opt/ml/processing/output", destination=out_uri)],
    )
    print(f"[job] done -> {out_uri}")


def main() -> None:
    if "--local" in sys.argv[1:]:
        run_local()
    else:
        run_job()


if __name__ == "__main__":
    main()
