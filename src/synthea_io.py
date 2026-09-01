"""Shared helpers: run Synthea and load its CSV export."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
JAR = ROOT / "synthea" / "synthea-with-dependencies.jar"
CSV_DIR = ROOT / "output" / "csv"

_HOMEBREW_JAVA = "/opt/homebrew/opt/openjdk@17/bin"


def _java() -> str:
    java = shutil.which("java")
    if java:
        return java
    cand = Path(_HOMEBREW_JAVA) / "java"
    if cand.exists():
        return str(cand)
    sys.exit("Java not found. `brew install openjdk@17` or add java to PATH.")


def run_synthea(n: int, state: str, seed: int, *, force: bool = False) -> None:
    """Generate `n` patients unless the CSV export already exists."""
    if CSV_DIR.joinpath("patients.csv").exists() and not force:
        print(f"[synthea] reusing existing export at {CSV_DIR}")
        return
    if not JAR.exists():
        sys.exit(f"Synthea jar missing at {JAR}. See README.")
    cmd = [
        _java(), "-jar", str(JAR),
        "-p", str(n), "-s", str(seed), "-cs", "12345",
        "--exporter.baseDirectory", str(ROOT / "output") + os.sep,
        "--exporter.fhir.export", "false",
        "--exporter.hospital.fhir.export", "false",
        "--exporter.practitioner.fhir.export", "false",
        "--exporter.csv.export", "true",
        "--exporter.text.export", "false",
        state,
    ]
    print("[synthea]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def load(*names: str) -> dict[str, pd.DataFrame]:
    """Load named CSVs (without extension) from the export."""
    out = {}
    for name in names:
        out[name] = pd.read_csv(CSV_DIR / f"{name}.csv", dtype=str, low_memory=False)
    return out
