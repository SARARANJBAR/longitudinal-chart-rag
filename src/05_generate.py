"""Stage 05 — grounded generation via Bedrock.

Given a query + retrieved chunks, ask Claude Haiku 4.5 (AnthropicBedrock, messages.create)
for a controlled/not-controlled answer plus the chunk_ids it relied on.

Prompt: system = "answer only from the provided encounters, cite chunk ids";
returns {answer, cited_chunk_ids, rationale}.
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
