# Longitudinal Chart RAG

Retrieval over a patient's whole chart, evaluated against FHIR ground truth.

- **Data:** Synthea synthetic patients, ~30-patient diabetic cohort
- **Measure:** CMS122 — was HbA1c controlled (≤ 9%) during a given measurement period?
- **Embedding:** `bge-small-en-v1.5` computed by a **transient SageMaker Processing Job**
  (reads corpus from S3 → writes FAISS index + parquet metadata back to S3 → terminates)
- **Retrieval:** FAISS dense + optional BM25 hybrid, top-k
- **Generation:** Claude Haiku 4.5 via Bedrock, grounded answer with chunk-ID citations
- **Eval:** `recall@{1,3,5}` + end-to-end answer accuracy vs FHIR-derived gold

See [`final_plan.md`](final_plan.md) for the full design, decisions table, and rationale.

## Layout

```
src/01_synthea.py        generate + filter to the CMS122 cohort
src/02_gold_labels.py    FHIR -> answer truth + gold chunk ids
src/03_embed_job.py      launch the SageMaker Processing Job
src/processing/embed.py  runs inside the job: chunk, embed, write faiss + parquet to S3
src/04_retrieve.py       load index from S3, dense + BM25, top-k
src/05_generate.py       Bedrock Haiku, grounded answer + citations
src/06_eval.py           recall@k, answer accuracy
src/07_ablations.py      run the ablation matrix, build the comparison table
notebooks/demo.ipynb     end-to-end walkthrough
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in bucket + role, then: set -a && source .env
```

Requires: AWS account (free tier), Bedrock Haiku model access enabled in `us-east-1`,
a SageMaker execution role, and Java (for Synthea).

## Status

Scaffolding. Build order follows `src/` numbering; see `final_plan.md` for the hour
estimate and fallbacks.
