import re
from langchain_core.prompts import ChatPromptTemplate
from app.llm import get_llm

llm = get_llm()

DIAGRAM_KEYWORDS = ["diagram", "flowchart", "chart", "visual", "illustrate"]
MERMAID_BLOCK_PATTERN = re.compile(r"```mermaid\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _extract_topic_from_query(query: str) -> str:
    lowered = query.lower()
    for phrase in ["flowchart for", "diagram for", "chart for", "visual for", "illustrate"]:
        if phrase in lowered:
            topic = lowered.split(phrase, 1)[1].strip()
            return topic or "the concept"

    cleaned = lowered
    for word in DIAGRAM_KEYWORDS + ["for", "of", "a", "an", "the"]:
        cleaned = cleaned.replace(word, " ")

    topic = " ".join(cleaned.split()).strip()
    return topic or "the concept"


def _safe_node_label(text: str) -> str:
    filtered = re.sub(r"[^a-zA-Z0-9 ]+", "", text).strip()
    if not filtered:
        return "Concept"
    words = filtered.split()
    return " ".join(words[:3])


def _fallback_diagram_response(query: str) -> str:
    topic = _extract_topic_from_query(query)
    topic_title = topic.title()

    core = _safe_node_label(topic_title)
    part_a = f"{core} Basics"
    part_b = f"{core} Process"
    part_c = f"{core} Outcomes"
    part_d = f"{core} Applications"

    return (
        f"Explanation: This flowchart gives a quick visual structure of {topic} and how its parts connect.\n\n"
        "Diagram:\n"
        "```mermaid\n"
        "graph TD\n"
        f"    {core} --> {part_a}\n"
        f"    {core} --> {part_b}\n"
        f"    {part_b} --> {part_c}\n"
        f"    {part_c} --> {part_d}\n"
        "```"
    )


def _ensure_mermaid_response(text: str, query: str) -> str:
    if MERMAID_BLOCK_PATTERN.search(text):
        return text.strip()
    return _fallback_diagram_response(query)


def is_diagram_query(query: str) -> bool:
    normalized = query.lower()
    return any(keyword in normalized for keyword in DIAGRAM_KEYWORDS)


def _normal_tutor_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template("""
    You are an expert academic tutor.

    Your goal is to teach with clarity, depth, and exam usefulness.
    Use simple language first, then build to deeper understanding.

    Response rules:
    1. Start with a 2-3 line direct answer.
    2. Then explain the concept step by step.
    3. Include one practical example (real-world or numerical when relevant).
    4. Add an "Exam Focus" section with 4-6 bullet points.
    5. Add a "Common Mistakes" section with 2-4 bullet points.
    6. End with a one-line memory tip.
    7. If the question is ambiguous, state your assumption clearly before answering.

    Keep the answer accurate, concise, and well-structured with clear headings.
    Avoid unnecessary jargon.

    Student question: {query}
    """)


def _diagram_tutor_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template("""
    You are an expert academic tutor.

    The student asked for a visual explanation. Produce output in this exact structure:

    Explanation: <short explanation>

    Diagram:
    ```mermaid
    graph TD
        NodeA --> NodeB
    ```

    Mermaid rules:
    - Use graph TD only.
    - Keep the diagram simple and readable.
    - Use clear node names.
    - Avoid special characters that can break Mermaid syntax.
    - Use at most 10 nodes.
    - Ensure diagram matches the user's topic exactly.
    - Return only one Mermaid code block.

    Student question: {query}
    """)

def tutor_agent_stream(query: str):
    prompt = _normal_tutor_prompt()
    chain = prompt | llm

    for chunk in chain.stream({"query": query}):
        # Chat model stream chunks expose incremental text in .content.
        content = getattr(chunk, "content", "")
        if isinstance(content, str) and content:
            yield content


def tutor_agent(query: str):
    if is_diagram_query(query):
        prompt = _diagram_tutor_prompt()
        chain = prompt | llm
        response = chain.invoke({"query": query})
        return _ensure_mermaid_response(response.content, query)

    return "".join(tutor_agent_stream(query))