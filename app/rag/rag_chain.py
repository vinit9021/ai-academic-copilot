from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_llm

llm = get_llm()


def _build_context(documents) -> str:
    context_parts = []

    for index, document in enumerate(documents, start=1):
        content = getattr(document, "page_content", "")
        if not isinstance(content, str):
            continue

        cleaned_content = content.strip()
        if not cleaned_content:
            continue

        context_parts.append(f"Chunk {index}:\n{cleaned_content}")

    return "\n\n".join(context_parts).strip()


def ask_rag(vector_store, query: str) -> str:
    normalized_query = str(query).strip()
    if not normalized_query or vector_store is None:
        return "Not in document"

    try:
        documents = vector_store.similarity_search(normalized_query, k=4)
    except Exception:
        return "Not in document"

    context = _build_context(documents)
    if not context:
        return "Not in document"

    prompt = ChatPromptTemplate.from_template(
        """
        You are a precise academic assistant.

        Use ONLY the provided context.
        If the answer is not clearly present in the context, reply exactly: "Not in document".

        Answer style:
        - Give a complete and helpful answer in 2-5 sentences when information is available.
        - Include relevant details (names, roles, dates, values, locations) from context.
        - If the user asks a short follow-up (for example: "only 1 internship?"), resolve it using context and answer directly.
        - Do not mention chunk numbers unless explicitly asked.
        - Do not invent or assume facts beyond context.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    try:
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": normalized_query})
        answer = getattr(response, "content", "")
        cleaned_answer = answer.strip() if isinstance(answer, str) else ""
        return cleaned_answer or "Not in document"
    except Exception:
        return "Not in document"
