"""Stage 02 — derive gold truth from the Synthea CSV export.

One query per (cohort patient, measurement year) where:
  - the patient is 18-75 at the end of the year and alive during it, and
  - at least one HbA1c (LOINC 4548-4) was recorded that year.

For each query:
  - answer:        most recent HbA1c that year <= 9.0  ->  "controlled" / "not controlled"
  - gold_chunk_ids: encounter_id(s) of that most-recent HbA1c observation

Years with no HbA1c are skipped: CMS122 would score them "poor control", but there
is no gold chunk to retrieve, so they don't belong in a retrieval eval.

NOTE (data finding): Synthea's synthetic diabetics are almost all well-controlled
(A1c rarely > 8%), so the binary controlled/not-controlled label has little
variance. Each query therefore also carries `hba1c_value` / `hba1c_date` so the
eval can score value extraction ("what was the most recent A1c that year?"),
which has real signal and still requires retrieving the right encounter.

Output: data/eval_set.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import config as cfg  # noqa: E402
from synthea_io import load  # noqa: E402


def main() -> None:
    roster = json.loads((ROOT / cfg.DATA_DIR / "cohort.json").read_text())
    d = load("observations", "encounters")
    obs, enc = d["observations"], d["encounters"]

    a1c = obs[obs.CODE == cfg.HBA1C_LOINC].copy()
    a1c["date"] = pd.to_datetime(a1c.DATE.str.slice(0, 10))
    a1c["year"] = a1c.date.dt.year
    a1c["value"] = pd.to_numeric(a1c.VALUE, errors="coerce")
    a1c = a1c.dropna(subset=["value"])

    enc_years = {}
    ed = pd.to_datetime(enc.START.str.slice(0, 10))
    for pid, yr in zip(enc.PATIENT, ed.dt.year):
        enc_years.setdefault(pid, set()).add(int(yr))

    rows = []
    for p in roster:
        pid = p["patient_id"]
        birth = pd.Timestamp(p["birthdate"])
        death = pd.Timestamp(p["deathdate"]) if p["deathdate"] else None
        pa1c = a1c[a1c.PATIENT == pid]
        for year in sorted(pa1c.year.unique()):
            age_end = int((pd.Timestamp(f"{year}-12-31") - birth).days // 365.25)
            if not (18 <= age_end <= 75):
                continue
            if death is not None and death.year < year:
                continue
            if year not in enc_years.get(pid, set()):
                continue
            yr = pa1c[pa1c.year == year]
            last_date = yr.date.max()
            latest = yr[yr.date == last_date]
            value = float(latest.value.iloc[0])
            controlled = value <= cfg.HBA1C_CONTROL_THRESHOLD
            rows.append({
                "query_id": f"{pid}:{year}",
                "patient_id": pid,
                "patient_name": p["name"],
                "year": int(year),
                "question": (
                    f"Was {p['name']}'s hemoglobin A1c controlled "
                    f"(≤ {cfg.HBA1C_CONTROL_THRESHOLD:.0f}%) during the {year} measurement period?"
                ),
                "answer": "controlled" if controlled else "not controlled",
                "hba1c_value": value,
                "hba1c_date": last_date.strftime("%Y-%m-%d"),
                "gold_chunk_ids": sorted(latest.ENCOUNTER.unique().tolist()),
            })

    out = ROOT / cfg.DATA_DIR / "eval_set.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = len(rows)
    ctrl = sum(r["answer"] == "controlled" for r in rows)
    print(f"[done] {n} queries across {len(roster)} patients -> {out}")
    print(f"       controlled: {ctrl}  |  not controlled: {n - ctrl}")
    print(f"       gold chunks per query: "
          f"{sum(len(r['gold_chunk_ids']) for r in rows) / n:.2f} avg")
    if n and min(ctrl, n - ctrl) / n < 0.15:
        print("       WARNING: control label is heavily imbalanced — "
              "prefer value-extraction scoring (hba1c_value) for the answer eval.")


if __name__ == "__main__":
    main()
