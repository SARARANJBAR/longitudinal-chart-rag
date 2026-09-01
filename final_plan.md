# Longitudinal Chart RAG — Final Plan

**Retrieval over a patient's whole chart, evaluated against structured ground truth.**
Synthea patients · CMS122 (HbA1c control) · SageMaker batch embedding · Claude on Bedrock.

| | |
|---|---|
| Core build | ~12.5 hrs |
| With AWS buffer | ~15.5 hrs |
| Expected AWS spend | < $5 (mostly Bedrock ≈ $2–3) |
| Deliverables | repo + README + architecture diagram + ablation table |
| Measure | 1 (CMS122) |
| Core ablations | 3 (+2 optional) |

---

## The retrieval problem

Pulling one lab value from one note isn't retrieval; the note fits in context.
Retrieval earns its place over a **longitudinal record**: Synthea emits years of
encounters per patient, and answering *"was this patient's HbA1c controlled during
the 2024 measurement period?"* means finding 2–3 documents out of ~200. Because
Synthea also emits a structured CSV export, the same data yields **free ground truth** for
both retrieval labels and the final answer.

---

## Design decisions from review

| Concern raised | Change |
|---|---|
| "Are you paying to keep a live embedding endpoint running?" | **No live endpoint.** Embedding is a one-time job → SageMaker **Processing Job** (transient, batch), terminates on completion. |
| "Use a transient batch workflow to show SageMaker" | Adopted as the core SageMaker artifact. |
| "BGE-M3 is overkill; use bge-small / MiniLM on CPU" | Switched to `BAAI/bge-small-en-v1.5` on `ml.m5.xlarge` (CPU). |
| "FAISS vs Chroma? flat file? S3? parquet?" | FAISS flat file + parquet metadata sidecar, both written to S3 by the job. |
| Stage 02 FHIR-traversal overrun risk | **Resolved** — use Synthea's **CSV export**, not FHIR. Notes are assembled per-encounter from `encounters.csv` + `observations.csv` + `conditions.csv` + `medications.csv`, so every note carries its real `encounter_id` and gold labels are a CSV filter, not a resource walk. |
| "Why pay Bedrock tokens when you have Claude Pro?" | Claude Pro is a chat UI with **no API access**. Programmatic generation needs Bedrock (or paid API). Bedrock Haiku for the full eval ≈ $2–3, within budget, and it demonstrates Bedrock. Kept. |
| "Streamlit instead of cloud?" | Skipped. One notebook cell / screenshot covers the demo; a UI adds hours and zero MLOps signal. |
| "Deploy a reranker endpoint, run eval, delete it" | Kept as an **optional** stretch ablation (real-time deploy → invoke → teardown lifecycle). |
| Project name | Kept short; rejected a longer buzzword-heavy alternative. |

---

## Pipeline

```
1  Synthea            2  SageMaker Processing Job          3  Retrieval (local)     4  Bedrock            5  Eval (local)
~30 diabetic pts  →   reads corpus + queries from S3   →   FAISS + optional BM25 →  Claude Haiku 4.5  →  vs CSV gold
per-encounter        bge-small-en-v1.5 on ml.m5.xlarge     load index from S3       grounded answer      recall@k +
notes (~6k)          embeds ~6k chunks + ~100 queries      top-k chunks             + chunk-ID cites     answer accuracy
                     writes .faiss + .parquet to S3
                     → terminates
```

Only managed step is **2** (transient) and the **4** API call. Everything else is local.

---

## Decisions

| Decision | Choice | Justification |
|---|---|---|
| SageMaker role | **Processing Job** — pull corpus + query set from S3, chunk, embed, write `index.faiss` + `chunks.parquet` back to S3, terminate | One-time embedding has no reason to be a 24/7 endpoint. Batch job = the honest MLOps pattern: containerized, S3 in/out, driven from the SDK, no idle cost. |
| Real-time endpoint | **Optional stretch** — deploy bge-small as a JumpStart real-time endpoint, serve query embeddings live during one eval run, delete in the same session | Shows deploy/invoke/teardown lifecycle for the résumé. Cut if it costs >1.5h or risks budget. |
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim) on `ml.m5.xlarge` (CPU) | Fast on CPU, ample for ~6k short clinical passages. BGE-M3 buys nothing here and needs a bigger instance. |
| Vector store | FAISS `IndexFlatIP`, single file; chunk text + metadata (patient_id, encounter_date, type) in a parquet sidecar; both in S3 | 6k vectors → exact flat search is instant; no need for IVF/HNSW. Parquet is the natural metadata format and enables the date-filter ablation. |
| Generation LLM | Claude Haiku 4.5 via Bedrock (`AnthropicBedrock`, `messages.create`) | No API access on Claude Pro. Grounded synthesis from 5 chunks — Haiku handles it; 3 ablations × ~100 queries ≈ $2–3. Same AWS auth as SageMaker. |
| Dev environment | Local Python + boto3/sagemaker SDK | No Studio lifecycle to manage; job and endpoint are a few SDK lines. |
| Note generation | Synthea JAR, local | Java, one command, no cloud. |
| Storage | S3 (free-tier 5 GB) for job I/O only; local disk for everything else | Dataset is a few hundred MB; S3 is just the job's input/output contract. |
| IAM | One execution role: SageMaker create-job/create-endpoint/invoke + `s3:*` on one bucket + `bedrock:InvokeModel` | Single role, no credential juggling. |
| Region | One region with both JumpStart bge-small and Bedrock Haiku access | Split regions doubles config for no gain. |
| Streamlit / UI | None | Adds hours, no MLOps signal. Demo = a notebook cell. |
| Cohort size | ~30 diabetic patients (overgenerate ~200, filter to CMS122 denominator) | Patient count matters only through the query set (below). |

---

## Eval design

**Query set (137):** for each cohort patient, one *"what was the most recent HbA1c
during measurement period Y?"* per year they have an HbA1c and sit in the CMS122
denominator (age 18–75, alive, a visit that year). This is the unit every ablation
delta is measured over. Actuals from Stage 01/02: 20 patients, 2,412 encounter
notes, 137 queries, 1.0 gold chunk each.

**Gold labels (from the CSV export):**
- *Answer truth* — the most recent HbA1c value in the period (30 distinct values
  across the 137 queries).
- *Gold chunks* — the encounter(s) containing that HbA1c observation.

**Metrics:** retrieval `recall@{1,3,5}`; end-to-end answer accuracy = generated
value within ±0.1% of truth; citation correctness (cited chunk ∈ gold) as a
secondary check. The CMS122 controlled/not-controlled flag is **reported but not
scored** — Synthea's synthetic diabetics are all controlled (A1c ≤ 7.6%), so the
binary label has no variance.

| Ablation | Question it answers | Status |
|---|---|---|
| Dense only, k=5 | Baseline | Core |
| + BM25 hybrid | Does lexical matching help on lab codes / numeric values? | Core |
| Date-filtered pre-retrieval | Does metadata filtering beat semantic ranking on temporal queries? | Core |
| Chunk = encounter vs. fixed 512 tokens | Does clinical structure beat naive chunking? (second embed pass) | Optional |
| + cross-encoder reranker on a SageMaker endpoint | Does reranking top-20→top-5 help? (doubles as the endpoint lifecycle demo) | Optional |

---

## Hours

| # | Stage | Hours |
|---|---|---:|
| 01 | Synthea run → filter to CMS122 diabetic cohort → per-encounter note text | 2.0 |
| 02 | Gold labels from CSV export (answer truth + gold chunk IDs) | 2.0 |
| 03 | Processing Job: script, container config, S3 wiring, run + verify index | 2.5 |
| 04 | Retrieval: load index from S3, FAISS + BM25, top-k function | 1.5 |
| 05 | Bedrock generation with chunk-ID citations | 1.5 |
| 06 | Eval harness: recall@k, end-to-end accuracy | 1.5 |
| 07 | Run 3 core ablations, build comparison table | 1.5 |
| 08 | README + architecture diagram | 1.0 |
| | **Core total** | **~12.5** |
| | *AWS friction buffer (Bedrock access, IAM, JumpStart config)* | *+3.0* |

---

## Risks and fallbacks

| Risk | Fallback |
|---|---|
| Cohort too small after CMS122 filter | Widen to 3 states / raise overgeneration count; or relax to ≥1 HbA1c-year per patient. |
| Bedrock Haiku access not granted in time | Run eval on a subset (~30 queries) while access clears; or drop to a local `transformers` small model for generation and note the limitation. |
| Processing Job / JumpStart quota friction | Fall back to embedding locally with the same `sentence-transformers` model; keep the job script in the repo as the intended path and document it. |
| Budget creep | Optional ablations and the real-time endpoint are the first things cut. |

---
*Clinical RAG · CMS122 · Synthea · SageMaker (transient batch) · Bedrock Haiku*
