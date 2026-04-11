import re
from langchain_core.prompts import ChatPromptTemplate
from app.llm import get_llm

llm = get_llm()


def evaluate_answer(topic: str, question: str, answer: str) -> str:
    prompt = ChatPromptTemplate.from_template("""
    You are an expert academic evaluator.

    Evaluate the student's answer based on:
    - Correctness
    - Completeness
    - Clarity
    - Relevance to the exact question and topic

    Topic: {topic}
    Question: {question}
    Student answer: {answer}

    Output format (follow exactly):
    SCORE: <integer from 0 to 10>
    VERDICT: <one short sentence>
    STRENGTHS:
    - <point 1>
    - <point 2>
    GAPS:
    - <point 1>
    - <point 2>
    IMPROVED_ANSWER:
    <a concise, high-quality answer the student should aim for>
    NEXT_STEP:
    <one actionable improvement tip>

    Rules:
    - Be strict but fair.
    - Do not invent facts not related to the topic/question.
    - Keep feedback concise and specific.
    - Return only the format above.
    """)

    chain = prompt | llm
    response = chain.invoke({"topic": topic, "question": question, "answer": answer})
    return response.content.strip()


def extract_score(evaluation_text: str) -> int:
    match = re.search(r"SCORE:\s*(\d{1,2})", evaluation_text, flags=re.IGNORECASE)
    if not match:
        return 0

    score = int(match.group(1))
    return max(0, min(10, score))