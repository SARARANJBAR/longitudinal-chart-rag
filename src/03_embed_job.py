"""Stage 03 — build the embedding index.

  python src/03_embed_job.py --local     chunk + embed locally -> artifacts/   (approach A: dev loop)
  python src/03_embed_job.py             launch the transient SageMaker Processing Job

The job runs src/processing/embed.py on a prebuilt PyTorch container (EMBED_INSTANCE in
config.yaml, CPU), uploads data/corpus/ + data/eval_set.jsonl as input, writes index.faiss
+ chunks.parquet + query_embeddings.parquet to s3://$RAG_S3_BUCKET/$PREFIX/output/, and
terminates. See TODO.md "S3 wiring" for current run status.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402

PROCESSING_DIR = ROOT / "src" / "processing"


def _build_source_dir() -> Path:
    """Stage a source_dir for the job: src/processing/ contents plus config.py +
    config.yaml copied alongside embed.py (it isn't shipped there locally — see
    embed.py's sys.path fallback). Kept out of git so the two stay single-sourced."""
    staging = Path(tempfile.mkdtemp(prefix="chart-rag-embed-src-"))
    shutil.copytree(PROCESSING_DIR, staging, dirs_exist_ok=True,
                     ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "config.py", staging / "config.py")
    shutil.copy2(ROOT / "config.yaml", staging / "config.yaml")
    return staging


def run_local() -> None:
    from processing.embed import run

    run(ROOT / cfg.DATA_DIR, ROOT / cfg.ARTIFACT_DIR)


def run_job() -> None:
    # sagemaker SDK v3 (pinned sagemaker>=3.0 in requirements.txt) restructured the
    # package: no more sagemaker.pytorch.PyTorchProcessor / sagemaker.processing top-level
    # module. Processing lives under sagemaker.core.processing, FrameworkProcessor now
    # needs an explicit image_uri (resolved via sagemaker.core.image_uris.retrieve), and
    # ProcessingInput/Output wrap a nested s3_input/s3_output object rather than
    # source=/destination= kwargs.
    from sagemaker.core.image_uris import retrieve as retrieve_image_uri
    from sagemaker.core.processing import (
        FrameworkProcessor, ProcessingInput, ProcessingOutput, ProcessingS3Input,
    )
    from sagemaker.core.shapes.shapes import ProcessingS3Output

    if not (cfg.S3_BUCKET and cfg.SM_EXECUTION_ROLE):
        sys.exit("Set RAG_S3_BUCKET and SM_EXECUTION_ROLE in .env first (see TODO.md).")

    out_uri = f"s3://{cfg.S3_BUCKET}/{cfg.S3_PREFIX}/output"
    # PyTorch 2.6/py312, not 2.3/py311: transformers==5.16.1 (pinned in
    # requirements.txt to match the local dev env) disables its own torch
    # integration unless torch>=2.5 is present, which crashed the 2.3 container
    # with `NameError: name 'nn' is not defined` on the first run of this job.
    image_uri = retrieve_image_uri(
        framework="pytorch", region=cfg.REGION, version="2.6", py_version="py312",
        image_scope="training", instance_type=cfg.EMBED_INSTANCE,
    )
    processor = FrameworkProcessor(
        image_uri=image_uri,
        role=cfg.SM_EXECUTION_ROLE,
        instance_type=cfg.EMBED_INSTANCE,
        instance_count=1,
        base_job_name="chart-rag-embed",
    )

    def s3_in(name: str, local_source: Path, container_path: str) -> ProcessingInput:
        return ProcessingInput(
            input_name=name,
            s3_input=ProcessingS3Input(
                s3_uri=str(local_source),  # local path -> SDK uploads to S3 itself
                local_path=container_path,
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_data_distribution_type="FullyReplicated",
            ),
        )

    source_dir = _build_source_dir()
    try:
        processor.run(
            code="embed.py",
            source_dir=str(source_dir),
            requirements="requirements.txt",
            inputs=[
                s3_in("corpus", ROOT / cfg.DATA_DIR / "corpus", "/opt/ml/processing/input/corpus"),
                s3_in("eval-set", ROOT / cfg.DATA_DIR / "eval_set.jsonl", "/opt/ml/processing/input"),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="artifacts",
                    s3_output=ProcessingS3Output(
                        s3_uri=out_uri, local_path="/opt/ml/processing/output",
                        s3_upload_mode="EndOfJob",
                    ),
                ),
            ],
        )
    finally:
        shutil.rmtree(source_dir, ignore_errors=True)
    print(f"[job] done -> {out_uri}")


def main() -> None:
    if "--local" in sys.argv[1:]:
        run_local()
    else:
        run_job()


if __name__ == "__main__":
    main()
