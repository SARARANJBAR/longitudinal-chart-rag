"""Stage 06 — eval harness.

For a given retrieval mode, over data/eval_set.jsonl:
  retrieval:  recall@{1,3,5}  (gold_chunk_ids vs retrieved encounter_ids)
  end-to-end: answer accuracy (hba1c_value within tolerance of gold, e.g. ±0.1%
              — exact definition TBD, see TODO.md Stage 06 decisions) [NOT YET —
              needs Stage 05 output]
  secondary:  citation correctness (cited_encounter_ids overlap gold), control_flag match
              [NOT YET — needs Stage 05 output]

Writes artifacts/results_<mode>.json.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402

# src/04_retrieve.py isn't a valid module name (leading digit) -> load by path.
_spec = importlib.util.spec_from_file_location("_retrieve", Path(__file__).with_name("04_retrieve.py"))
_retrieve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_retrieve)

MODES = ("dense", "hybrid", "date_filtered")
KS = (1, 3, 5)


def recall_at_k(retriever, queries: list[dict], mode: str, k: int) -> tuple[float, list[dict]]:
    hits = 0
    rows = []
    for q in queries:
        gold = set(q["gold_chunk_ids"])
        got = retriever.retrieve(q["query_id"], mode, k)
        hit = bool(gold & set(got))
        hits += hit
        rows.append({"query_id": q["query_id"], "hit": hit, "retrieved": got})
    return hits / len(queries), rows


def main() -> None:
    retriever = _retrieve.get_retriever()
    queries = _retrieve._load_eval_set()
    out_dir = ROOT / cfg.ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] {len(queries)} queries, modes={MODES}, k={KS}\n")
    print(f"{'mode':14} " + "  ".join(f"recall@{k}" for k in KS))
    for mode in MODES:
        per_k = {}
        detail_at_5 = None
        for k in KS:
            recall, rows = recall_at_k(retriever, queries, mode, k)
            per_k[k] = recall
            if k == 5:
                detail_at_5 = rows
        print(f"{mode:14} " + "  ".join(f"{per_k[k]:.3f}   " for k in KS))

        out = {
            "mode": mode,
            "n_queries": len(queries),
            "recall_at_k": per_k,
            "per_query_at_5": detail_at_5,
        }
        (out_dir / f"results_{mode}.json").write_text(json.dumps(out, indent=2))
    print(f"\n[eval] wrote artifacts/results_<mode>.json for {', '.join(MODES)}")
    print("[eval] answer accuracy / citation correctness not scored — needs Stage 05 output.")


if __name__ == "__main__":
    main()
