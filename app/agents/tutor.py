from langchain_core.prompts import ChatPromptTemplate
from app.llm import get_llm

llm = get_llm()

def tutor_agent_stream(query: str):
    prompt = ChatPromptTemplate.from_template("""
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

    chain = prompt | llm

    for chunk in chain.stream({"query": query}):
        # Chat model stream chunks expose incremental text in .content.
        content = getattr(chunk, "content", "")
        if isinstance(content, str) and content:
            yield content


def tutor_agent(query: str):
    return "".join(tutor_agent_stream(query))