from app.llm import get_llm

llm = get_llm()

def refine_query(query: str) -> str:
    prompt = f"""
    You are a precise spelling and grammar refiner for student queries.

    Task:
    - Fix spelling mistakes in the query.
    - Fix obvious grammar/punctuation issues only when needed for clarity.

    Strict rules:
    - Preserve the original meaning exactly.
    - Do not add new information.
    - Do not remove important words.
    - Keep technical terms, names, formulas, and abbreviations unchanged unless clearly misspelled.
    - If the query is already correct, return it unchanged.
    - Return exactly one refined query line and nothing else.

    Query: {query}

    Refined query:
    """

    response = llm.invoke(prompt)
    return response.content.strip()