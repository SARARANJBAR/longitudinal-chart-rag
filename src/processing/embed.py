"""Chunk + embed the corpus. Runs INSIDE the SageMaker Processing Job container,
and is importable for the local build (Stage 03, approach A).

Input   <input>/corpus/<patient_id>/<encounter_id>.txt   per-encounter notes
        <input>/eval_set.jsonl                            queries
Output  <output>/index.faiss              IndexFlatIP over normalized chunk vectors
        <output>/chunks.parquet           chunk_id, encounter_id, patient_id,
                                           encounter_date, visit_type, text  (row order == faiss id)
        <output>/query_embeddings.parquet query_id, embedding

Chunking (config.CHUNK_STRATEGY):
  "section"  — keep the note header on every chunk, then pack the body section by
               section, starting a new chunk at each section header or when the
               next line would exceed MAX_CHUNK_TOKENS. No truncation.
  "fixed512" — naive sliding window over the whole note (optional ablation).

In the container SageMaker maps <output> to S3 automatically.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
# In the SageMaker container, source_dir is flattened so embed.py sits at the code
# root with config.py/config.yaml copied alongside it (see 03_embed_job.py) rather
# than three levels up as in the local src/processing/ layout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg  # noqa: E402

_DATE_RE = re.compile(r"^Date:\s*(\S+)", re.M)
_TYPE_RE = re.compile(r"^Visit type:\s*(.+)$", re.M)


def _load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg.EMBED_MODEL)


def _ntok(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=True))


def _split_header_body(note: str) -> tuple[str, list[str]]:
    """Header block = everything up to the first blank line; body = the rest, by line."""
    parts = note.split("\n\n", 1)
    header = parts[0].strip()
    body = parts[1].splitlines() if len(parts) > 1 else []
    return header, [ln for ln in body if ln.strip()]


def _is_section_header(line: str) -> bool:
    return not line.startswith(" ") and line.rstrip().endswith(":")


def chunk_section(note: str, tok, max_tok: int) -> list[str]:
    header, body = _split_header_body(note)
    if not body:
        return [header]
    chunks, cur = [], []

    def flush():
        if cur:
            chunks.append(header + "\n\n" + "\n".join(cur))
            cur.clear()

    for line in body:
        if _is_section_header(line) and cur:
            flush()
        trial = header + "\n\n" + "\n".join(cur + [line])
        if cur and _ntok(tok, trial) > max_tok:
            flush()
        cur.append(line)
    flush()
    return chunks


def chunk_fixed(note: str, tok, max_tok: int, overlap: int = 64) -> list[str]:
    ids = tok.encode(note, add_special_tokens=False)
    if len(ids) <= max_tok:
        return [note]
    step = max_tok - overlap
    return [tok.decode(ids[i : i + max_tok]) for i in range(0, len(ids), step)]


def build_chunks(input_dir: Path, tok) -> list[dict]:
    strategy = cfg.CHUNK_STRATEGY
    max_tok = cfg.MAX_CHUNK_TOKENS
    splitter = chunk_fixed if strategy == "fixed512" else chunk_section

    rows = []
    for note_path in sorted((input_dir / "corpus").glob("*/*.txt")):
        text = note_path.read_text()
        patient_id = note_path.parent.name
        encounter_id = note_path.stem
        date = m.group(1) if (m := _DATE_RE.search(text)) else ""
        visit_type = m.group(1).strip() if (m := _TYPE_RE.search(text)) else ""
        for i, chunk_text in enumerate(splitter(text, tok, max_tok)):
            rows.append({
                "chunk_id": f"{encounter_id}#{i:02d}",
                "encounter_id": encounter_id,
                "patient_id": patient_id,
                "encounter_date": date,
                "visit_type": visit_type,
                "text": chunk_text,
            })
    return rows


def run(input_dir, output_dir) -> None:
    import faiss
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tok = _load_tokenizer()
    chunks = build_chunks(input_dir, tok)
    queries = [json.loads(l) for l in (input_dir / "eval_set.jsonl").read_text().splitlines() if l.strip()]

    over = sum(_ntok(tok, c["text"]) > cfg.MAX_CHUNK_TOKENS for c in chunks)
    print(f"[embed] {len(chunks)} chunks from {len({c['encounter_id'] for c in chunks})} notes "
          f"({len(chunks) / len({c['encounter_id'] for c in chunks}):.2f}/note); {over} over {cfg.MAX_CHUNK_TOKENS} tok")

    model = SentenceTransformer(cfg.EMBED_MODEL)
    chunk_vecs = model.encode([c["text"] for c in chunks], batch_size=64,
                              normalize_embeddings=True, show_progress_bar=True).astype("float32")
    query_vecs = model.encode([cfg.QUERY_PREFIX + q["question"] for q in queries], batch_size=64,
                              normalize_embeddings=True, show_progress_bar=True).astype("float32")

    index = faiss.IndexFlatIP(chunk_vecs.shape[1])
    index.add(chunk_vecs)
    faiss.write_index(index, str(output_dir / "index.faiss"))
    pd.DataFrame(chunks).to_parquet(output_dir / "chunks.parquet", index=False)
    pd.DataFrame({"query_id": [q["query_id"] for q in queries],
                  "embedding": list(query_vecs)}).to_parquet(output_dir / "query_embeddings.parquet", index=False)
    print(f"[embed] wrote index.faiss ({index.ntotal} vecs, dim {chunk_vecs.shape[1]}), "
          f"chunks.parquet, query_embeddings.parquet -> {output_dir}")


def main() -> None:
    default_in, default_out = "/opt/ml/processing/input", "/opt/ml/processing/output"
    args = sys.argv[1:]
    run(args[0] if args else default_in, args[1] if len(args) > 1 else default_out)


if __name__ == "__main__":
    main()
