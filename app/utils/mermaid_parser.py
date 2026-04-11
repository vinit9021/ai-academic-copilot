import re


MERMAID_BLOCK_PATTERN = re.compile(
    r"```mermaid\s*(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_mermaid_block(text: str) -> tuple[str, str | None]:
    match = MERMAID_BLOCK_PATTERN.search(text)
    if not match:
        return text.strip(), None

    mermaid_code = match.group(1).strip()

    before = text[:match.start()].strip()
    after = text[match.end():].strip()

    explanation_parts = [part for part in [before, after] if part]
    explanation_text = "\n\n".join(explanation_parts).strip()

    if explanation_text.lower().startswith("diagram:"):
        explanation_text = explanation_text[len("diagram:"):].strip()

    return explanation_text, mermaid_code
