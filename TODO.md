# TODO

## Done
- [x] Env: OpenJDK 17 (`/opt/homebrew/opt/openjdk@17/bin`), py3.12 venv at `.venv`, deps installed
- [x] Stage 01 — Synthea CSV cohort: 20 type-2 diabetic patients, 2,412 per-encounter notes → `data/corpus/`, `data/cohort.json`
- [x] Stage 02 — gold labels: 137 queries (patient × measurement year) → `data/eval_set.jsonl`
      - Reframed to **value extraction** (Option A, approved): primary target `hba1c_value` = most recent A1c that year (30 distinct values). `control_flag` reported-only (all 137 "controlled" — no variance).

## Stage 03 — embedding → FAISS index. LOCAL BUILD DONE (2026-09-01); cloud run deferred.

Result: `artifacts/index.faiss` (7,600 chunks, 384-dim), `chunks.parquet`, `query_embeddings.parquet`.
Sanity checks pass — all 137 gold encounters indexed; A1c line + value co-located in every gold chunk;
3.15 chunks/note; 0 chunks over 512 tokens.

### DECISIONS (approved 2026-09-01):

1. **Build approach: A — local first.** Build the FAISS index locally with `sentence-transformers`
   (`bge-small-en-v1.5`, free, seconds per run), get stages 04–06 working against it, then run the
   SageMaker Processing Job **once** as the production artifact.
   - *Rationale:* same artifact either way (transient batch job, terminates); local iteration is
     seconds vs ~5–10 min/cycle and $0 vs cents; unblocked now (no AWS account yet — see below).
   - *Constraint:* the job (`03_embed_job.py` + `processing/embed.py`) must be written for real and
     run once; the Stage 04 index loader must read S3 **or** local via a config switch, so the
     "load from S3" contract is genuinely exercised.

2. **Chunking: B — section-header split, cap ~512 tokens.** ~3k chunks. Each chunk keeps its
   `encounter_id`; Stage 06 recall counts "any chunk of the gold encounter" as a hit.
   - *Rationale (from corpus measurement):* notes are bimodal — p50 221 tok, but 32% exceed 512 and
     mean is 544. The A1c line sits at a **median token offset of 662**; encounter-as-one-chunk
     with 512 truncation would drop the A1c value from **~51% of A1c-bearing gold chunks**. That is
     a broken baseline, not an acceptable simplification. Truncation-free splitting is now the
     baseline; the optional ablation is reframed to "section-aware split vs. naive fixed-512
     sliding window".

3. **Instance: A — `ml.t3.medium`** (~$0.01–0.03/job; no Processing free tier). `config.EMBED_INSTANCE`
   to be updated from `ml.m5.xlarge`.

### Stage 03 todo items
- [x] `config.py`: `EMBED_INSTANCE = "ml.t3.medium"`; added `MAX_CHUNK_TOKENS`, `QUERY_PREFIX`, `INDEX_SOURCE`.
- [x] `src/processing/embed.py` — section splitter (`chunk_section`) + `chunk_fixed` for the ablation; writes the 3 output files; runs in-container (`/opt/ml/processing/...`) or local via argv. `+ src/processing/requirements.txt` for the container.
- [x] `src/03_embed_job.py`: `--local` runs `embed.run()` in-process; default builds the `PyTorchProcessor` job (cloud path written, not yet run).
- [x] `chunk_id` scheme: `{encounter_id}#{nn}`.
- [x] Local run + sanity checks (see above).
- [ ] **Stage 04 must honor `INDEX_SOURCE`** — load `artifacts/` (local) or `s3://.../output/` so the S3 contract is real.
- [ ] Update `final_plan.md` Eval-design + optional-ablation wording to chunking B (defer to Stage 07/08).
- [ ] Deferred: the one real SageMaker job run + verify `framework_version`/`py_version` against a t3.medium-supported PyTorch container (needs AWS prereqs below).

## AWS prerequisites (needed before the cloud run — set in `.env`, see `.env.example`)
- [ ] **Not set up yet.** AWS account + S3 bucket + SageMaker execution role do not exist.
- [ ] S3 bucket name (free tier, 5 GB) → `RAG_S3_BUCKET`
- [ ] SageMaker execution role ARN → `SM_EXECUTION_ROLE`
- [ ] Bedrock **Claude Haiku 4.5** access in **us-east-1** — request early (longest lead time), needed for Stage 05 run.
- [ ] Confirm region = us-east-1

## Reminders
- Rule: get explicit OK on every stage decision before implementing.
- Rule: when I say "status", do exactly this, in order: (1) orient (re-read README, final_plan.md, TODO.md), (2) "Stage status vs. final_plan.md" table, (3) code review of src/ adherence to the plan, (4) next immediate step (cleanup or next stage). Report as a table. Do nothing else until I OK it.
- Commit/push only when asked. Currently on branch `stage-01-synthea`, pushed: `main` only.
- Plan: `final_plan.md`. Config: `config.py`.
