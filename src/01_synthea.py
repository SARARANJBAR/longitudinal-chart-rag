"""Stage 01 — generate Synthea patients and build the CMS122 diabetic cohort.

Runs Synthea (CSV export) with overgeneration, then keeps patients who:
  - have a type-2 diabetes condition (SNOMED 44054006), and
  - have >= MIN_HBA1C_YEARS distinct calendar years with an HbA1c (LOINC 4548-4).

Notes are assembled PER ENCOUNTER from the CSV export so each note keeps its real
encounter_id (used as the gold-chunk id in Stage 02).

Output:
  data/corpus/<patient_id>/<encounter_id>.txt   per-encounter note text
  data/cohort.json                              cohort roster + stats
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

import config as cfg
from synthea_io import load, run_synthea

CORPUS = ROOT / cfg.DATA_DIR / "corpus"


def _age(birthdate: str, on: pd.Timestamp) -> int:
    b = pd.Timestamp(birthdate)
    return int((on - b).days // 365.25)


def select_cohort(cond: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    dm2 = set(cond.loc[cond.CODE == cfg.DIABETES_SNOMED, "PATIENT"])
    a1c = obs[obs.CODE == cfg.HBA1C_LOINC].copy()
    a1c["year"] = a1c.DATE.str.slice(0, 4)
    years = a1c.groupby("PATIENT").year.nunique()
    keep = sorted(p for p in dm2 if years.get(p, 0) >= cfg.MIN_HBA1C_YEARS)
    return keep, a1c


def render_note(pat: pd.Series, enc: pd.Series, *, obs, cond, meds, procs) -> str:
    date = enc.START[:10]
    L = [
        "CLINICAL ENCOUNTER NOTE",
        f"Patient: {pat.FIRST} {pat.LAST} (sex {pat.GENDER}, DOB {pat.BIRTHDATE})",
        f"Patient ID: {pat.Id}",
        f"Encounter ID: {enc.Id}",
        f"Date: {date}   Age at visit: {_age(pat.BIRTHDATE, pd.Timestamp(date))}",
        f"Visit type: {enc.ENCOUNTERCLASS} — {enc.DESCRIPTION}",
    ]
    if isinstance(enc.REASONDESCRIPTION, str) and enc.REASONDESCRIPTION:
        L.append(f"Reason for visit: {enc.REASONDESCRIPTION}")

    labs = obs[obs.ENCOUNTER == enc.Id]
    if len(labs):
        L.append("\nVital signs and laboratory results:")
        for _, o in labs.iterrows():
            L.append(f"  - {o.DESCRIPTION}: {o.VALUE} {o.UNITS if isinstance(o.UNITS, str) else ''}".rstrip())

    dx = cond[cond.ENCOUNTER == enc.Id]
    if len(dx):
        L.append("\nDiagnoses recorded this visit:")
        for _, c in dx.iterrows():
            L.append(f"  - {c.DESCRIPTION}")

    rx = meds[meds.ENCOUNTER == enc.Id]
    if len(rx):
        L.append("\nMedications this visit:")
        for _, m in rx.iterrows():
            reason = f" (for {m.REASONDESCRIPTION})" if isinstance(m.REASONDESCRIPTION, str) and m.REASONDESCRIPTION else ""
            L.append(f"  - {m.DESCRIPTION}{reason}")

    pr = procs[procs.ENCOUNTER == enc.Id]
    if len(pr):
        L.append("\nProcedures this visit:")
        for _, p in pr.iterrows():
            L.append(f"  - {p.DESCRIPTION}")

    return "\n".join(L) + "\n"


def main() -> None:
    run_synthea(cfg.SYNTHEA_OVERGEN, cfg.SYNTHEA_STATE, cfg.SYNTHEA_SEED)
    d = load("patients", "encounters", "observations", "conditions", "medications", "procedures")
    patients, enc, obs = d["patients"], d["encounters"], d["observations"]
    cond, meds, procs = d["conditions"], d["medications"], d["procedures"]

    cohort_ids, a1c = select_cohort(cond, obs)
    print(f"[cohort] {len(cohort_ids)} type-2 diabetic patients with >= {cfg.MIN_HBA1C_YEARS} HbA1c years")
    if not cohort_ids:
        sys.exit("Empty cohort — raise SYNTHEA_OVERGEN or lower MIN_HBA1C_YEARS.")

    pat_by_id = patients.set_index("Id")
    roster = []
    n_notes = 0
    for pid in cohort_ids:
        pat = pat_by_id.loc[pid].rename_axis(None)
        pat = pat.copy()
        pat["Id"] = pid
        penc = enc[enc.PATIENT == pid].sort_values("START")
        pobs, pcond = obs[obs.PATIENT == pid], cond[cond.PATIENT == pid]
        pmeds, pprocs = meds[meds.PATIENT == pid], procs[procs.PATIENT == pid]

        pdir = CORPUS / pid
        pdir.mkdir(parents=True, exist_ok=True)
        for _, e in penc.iterrows():
            (pdir / f"{e.Id}.txt").write_text(
                render_note(pat, e, obs=pobs, cond=pcond, meds=pmeds, procs=pprocs)
            )
            n_notes += 1

        pyears = sorted(a1c.loc[a1c.PATIENT == pid, "year"].unique().tolist())
        roster.append({
            "patient_id": pid,
            "name": f"{pat.FIRST} {pat.LAST}",
            "birthdate": pat.BIRTHDATE,
            "deathdate": pat.DEATHDATE if isinstance(pat.DEATHDATE, str) else None,
            "gender": pat.GENDER,
            "n_encounters": int(len(penc)),
            "hba1c_years": pyears,
        })

    out = ROOT / cfg.DATA_DIR / "cohort.json"
    out.write_text(json.dumps(roster, indent=2))
    print(f"[done] {n_notes} encounter notes across {len(roster)} patients -> {CORPUS}")
    print(f"[done] roster -> {out}")


if __name__ == "__main__":
    main()
