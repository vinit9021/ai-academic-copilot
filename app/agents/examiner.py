from langchain_core.prompts import ChatPromptTemplate
from app.llm import get_llm

llm = get_llm()

def examiner_agent(topic: str):
    prompt = ChatPromptTemplate.from_template("""
    You are an expert exam setter.

    Create exactly 5 exam questions on the user's exact topic.
    Topic: {topic}

    Important constraints:
    - Stay strictly within the exact topic the user provided.
    - Do not broaden, narrow, rename, or replace the topic.
    - Do not add unrelated subtopics, background, or filler text.
    - Keep the wording aligned to the topic as written, including key scope words.

    Question mix:
    - 2 conceptual questions
    - 2 application-based questions
    - 1 challenging, interview-level question

    Quality rules:
    - Make the questions clear, specific, and suitable for college exams.
    - Avoid vague wording and avoid duplicate angles.
    - Ensure each question is genuinely about the same topic.

    Output only the 5 questions, one per line.
    """)

    chain = prompt | llm

    response = chain.invoke({"topic": topic})

    return response.content