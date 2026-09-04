"""Stage 05 — grounded generation via Bedrock.

Given a query + retrieved chunks, ask Claude Haiku 4.5 (AnthropicBedrock, messages.create)
for the hba1c_value (primary target) plus the encounter_ids it relied on. control_flag is
reported only (see Stage 02 note — label is degenerate in this cohort).

Prompt: system = "answer only from the provided encounters, cite encounter ids";
returns {hba1c_value, cited_encounter_ids, rationale}.
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
