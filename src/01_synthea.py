"""Stage 01 — generate Synthea patients and filter to the CMS122 diabetic cohort.

Output:
  data/corpus/<patient_id>/<encounter_date>.txt   per-encounter note text
  data/cohort.json                                 patient ids + measurement years

Approach:
  - Run Synthea with text export on, overgenerate ~200 patients.
  - Keep patients with a diabetes condition + >=1 HbA1c observation, cap at ~30.
  - Emit one note file per encounter (from Synthea's text export).

Fallback (see final_plan.md): if the cohort filter is fiddly, hardcode 5 patients.
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
