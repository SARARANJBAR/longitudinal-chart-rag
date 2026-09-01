"""Stage 04 — retrieval.

Loads index.faiss + chunks.parquet + query_embeddings.parquet (local artifacts/,
or downloaded from s3://$BUCKET/$PREFIX/output/ when config.INDEX_SOURCE == "s3")
and serves three modes, each returning a ranked list of **encounter_ids**
(chunks are deduped to their encounter, best rank kept):

  dense(query_id, k)          FAISS IndexFlatIP over the query embedding
  hybrid(query_id, k)         dense + BM25 over chunk text, fused with RRF
  date_filtered(query_id, k)  restrict to encounters in the query's year, then dense

Query vectors already carry config.QUERY_PREFIX (baked in at Stage 03).
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as cfg  # noqa: E402

_ARTIFACTS = ("index.faiss", "chunks.parquet", "query_embeddings.parquet")
_TOKEN = re.compile(r"\d+\.\d+|[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _resolve_artifact_dir(source: str) -> Path:
    local = ROOT / cfg.ARTIFACT_DIR
    if source == "local":
        return local
    if not cfg.S3_BUCKET:
        sys.exit("INDEX_SOURCE=s3 but RAG_S3_BUCKET is unset (see .env.example).")
    import boto3

    cache = local / "_s3cache"
    cache.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name=cfg.REGION)
    for name in _ARTIFACTS:
        dst = cache / name
        if not dst.exists():
            s3.download_file(cfg.S3_BUCKET, f"{cfg.S3_PREFIX}/output/{name}", str(dst))
    return cache


class Retriever:
    def __init__(self, source: str | None = None):
        import faiss
        from rank_bm25 import BM25Okapi

        adir = _resolve_artifact_dir(source or cfg.INDEX_SOURCE)
        self.index = faiss.read_index(str(adir / "index.faiss"))
        self.vecs = self.index.reconstruct_n(0, self.index.ntotal)  # (n_chunks, dim)

        chunks = pd.read_parquet(adir / "chunks.parquet")
        self.enc_ids = chunks.encounter_id.to_numpy()
        self.texts = chunks.text.tolist()
        self.years = chunks.encounter_date.str.slice(0, 4).to_numpy()
        self.bm25 = BM25Okapi([_tokenize(t) for t in self.texts])

        qe = pd.read_parquet(adir / "query_embeddings.parquet")
        self.qvec = {qid: np.asarray(v, dtype="float32")
                     for qid, v in zip(qe.query_id, qe.embedding)}
        self.query = {q["query_id"]: q for q in _load_eval_set()}

    # --- ranking helpers -------------------------------------------------
    def _dense_ranked(self, qid: str, depth: int) -> list[int]:
        q = self.qvec[qid][None, :]
        _, idx = self.index.search(q, depth)
        return idx[0].tolist()

    def _bm25_ranked(self, qid: str, depth: int) -> list[int]:
        scores = self.bm25.get_scores(_tokenize(self.query[qid]["question"]))
        return np.argsort(scores)[::-1][:depth].tolist()

    def _to_encounters(self, chunk_idxs: list[int], k: int) -> list[str]:
        out: list[str] = []
        for ci in chunk_idxs:
            e = self.enc_ids[ci]
            if e not in out:
                out.append(e)
            if len(out) == k:
                break
        return out

    # --- modes ---------------------------------------------------------
    def dense(self, qid: str, k: int = cfg.TOP_K) -> list[str]:
        return self._to_encounters(self._dense_ranked(qid, cfg.CANDIDATE_DEPTH), k)

    def hybrid(self, qid: str, k: int = cfg.TOP_K) -> list[str]:
        lists = [self._dense_ranked(qid, cfg.CANDIDATE_DEPTH),
                 self._bm25_ranked(qid, cfg.CANDIDATE_DEPTH)]
        fused = _rrf(lists, cfg.FUSION_RRF_K)
        return self._to_encounters(fused, k)

    def date_filtered(self, qid: str, k: int = cfg.TOP_K) -> list[str]:
        year = str(self.query[qid]["year"])
        pool = np.where(self.years == year)[0]
        if len(pool) == 0:
            return []
        scores = self.vecs[pool] @ self.qvec[qid]
        ranked = pool[np.argsort(scores)[::-1]].tolist()
        return self._to_encounters(ranked, k)

    def retrieve(self, qid: str, mode: str = "dense", k: int = cfg.TOP_K) -> list[str]:
        return {"dense": self.dense, "hybrid": self.hybrid,
                "date_filtered": self.date_filtered}[mode](qid, k)


def _rrf(rank_lists: list[list[int]], k: int) -> list[int]:
    score: dict[int, float] = defaultdict(float)
    for rl in rank_lists:
        for rank, idx in enumerate(rl):
            score[idx] += 1.0 / (k + rank + 1)
    return sorted(score, key=score.get, reverse=True)


def _load_eval_set() -> list[dict]:
    path = ROOT / cfg.DATA_DIR / "eval_set.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()


def main() -> None:
    r = get_retriever()
    for q in _load_eval_set()[:3]:
        qid, gold = q["query_id"], set(q["gold_chunk_ids"])
        print(f"\n{qid}  gold={sorted(gold)}")
        for mode in ("dense", "hybrid", "date_filtered"):
            got = r.retrieve(qid, mode)
            hit = "HIT" if gold & set(got) else "miss"
            print(f"  {mode:14} [{hit}] {got}")


if __name__ == "__main__":
    main()
