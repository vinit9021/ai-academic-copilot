import os

from app.router import route_query
from app.agents.tutor import tutor_agent_stream, tutor_agent, is_diagram_query
from app.agents.examiner import examiner_agent
from app.agents.evaluator import evaluate_answer, extract_score
from app.memory.weakness_tracker import update_weakness, generate_report
from app.rag import ask_rag, create_vector_store, load_file
from app.utils.mermaid_parser import extract_mermaid_block


FILE_TRIGGER_KEYWORDS = ("pdf", "document", "notes", "read file", "upload", "file")
DATA_STORAGE_DIR = "data"
_CURRENT_FILE_SOURCE_PATH = None
_CURRENT_FILE_VECTOR_STORE = None


def _normalize_question_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""

    line = line.lstrip("-*").strip()

    lowered = line.lower()
    if lowered.startswith("q") and len(line) > 2 and line[1].isdigit():
        line = line[2:].lstrip(".:)- ")
    elif line[:1].isdigit():
        idx = 0
        while idx < len(line) and line[idx].isdigit():
            idx += 1
        line = line[idx:].lstrip(".:)- ")

    return line.strip()


def _extract_questions(text: str) -> list[str]:
    questions = []
    for raw_line in text.splitlines():
        cleaned = _normalize_question_line(raw_line)
        if cleaned and cleaned.endswith("?"):
            questions.append(cleaned)

    if not questions:
        for raw_line in text.splitlines():
            cleaned = _normalize_question_line(raw_line)
            if cleaned:
                questions.append(cleaned)

    return questions[:5]


def _is_file_query(query: str) -> bool:
    normalized_query = query.lower()
    return any(keyword in normalized_query for keyword in FILE_TRIGGER_KEYWORDS)


def _get_file_vector_store(file_path: str):
    global _CURRENT_FILE_SOURCE_PATH, _CURRENT_FILE_VECTOR_STORE

    raw_path = os.path.expanduser(file_path.strip())
    if not raw_path:
        return None, "Invalid file path."

    if os.path.isabs(raw_path):
        normalized_path = os.path.abspath(raw_path)
    elif raw_path.startswith(f"{DATA_STORAGE_DIR}{os.sep}") or raw_path.startswith("data/") or raw_path.startswith("data\\"):
        normalized_path = os.path.abspath(raw_path)
    else:
        normalized_path = os.path.abspath(os.path.join(DATA_STORAGE_DIR, raw_path))

    if not os.path.exists(normalized_path):
        return None, "File not found."

    if (
        _CURRENT_FILE_VECTOR_STORE is not None
        and _CURRENT_FILE_SOURCE_PATH == normalized_path
    ):
        return _CURRENT_FILE_VECTOR_STORE, None

    try:
        documents = load_file(normalized_path)
    except FileNotFoundError:
        return None, "File not found."
    except ValueError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"Could not load file: {exc}"

    if not documents:
        return None, "Could not load file or the file is empty."

    vector_store = create_vector_store(documents)
    if vector_store is None:
        return None, "Could not build vector store."

    _CURRENT_FILE_SOURCE_PATH = normalized_path
    _CURRENT_FILE_VECTOR_STORE = vector_store
    return vector_store, None


def _run_file_rag_session():
    file_path = input("Enter file path inside data/ (example: data/sample.pdf): ").strip()
    vector_store, error_message = _get_file_vector_store(file_path)

    if error_message:
        print(error_message)
        return

    print("File loaded successfully. You can now ask questions.")
    print("\nDocument Q&A Session:\n")

    while True:
        document_query = input("Ask question from document (type 'exit' to stop): ").strip()

        if document_query.lower() == "exit":
            break

        if not document_query:
            continue

        answer = ask_rag(vector_store, document_query)
        print("\nAnswer:")
        if answer.strip().lower() == "not in document":
            print("Answer not found in document")
        else:
            print(answer)
        print()


def main():
    while True:
        query = input("Ask (type 'exit' to quit): ").strip()

        if not query:
            continue

        if query.lower() == "exit":
            print("Goodbye 👋")
            break

        if _is_file_query(query):
            _run_file_rag_session()
            print()
            continue

        route = route_query(query)

        print("\nAnswer:")

        # 🧠 Tutor Agent (Streaming)
        if route == "tutor":
            if is_diagram_query(query):
                full_response = tutor_agent(query)
                explanation_text, mermaid_code = extract_mermaid_block(full_response)

                if explanation_text:
                    print(explanation_text)

                if mermaid_code:
                    print("\nGenerated Diagram (Mermaid):")
                    print(mermaid_code)
                else:
                    print(full_response)
            else:
                pending = ""

                for piece in tutor_agent_stream(query):
                    pending += piece
                    while "\n" in pending:
                        line, pending = pending.split("\n", 1)
                        print(line)

                if pending:
                    print(pending)

        # 🧪 Examiner Agent + Evaluation + Tracking
        elif route == "examiner":
            topic = input("Exam topic: ").strip() or query
            response = examiner_agent(topic)
            questions = _extract_questions(response)

            if not questions:
                print("Could not generate valid questions. Try again.")
                print()
                continue

            print("\nPractice Session:\n")

            total_score = 0
            attempted = 0

            for index, question in enumerate(questions, start=1):
                print(f"Q{index}. {question}")
                user_answer = input("Your answer: ").strip()

                if not user_answer:
                    user_answer = "No answer provided."

                evaluation = evaluate_answer(
                    topic=topic,
                    question=question,
                    answer=user_answer
                )

                score = extract_score(evaluation)

                # 🔥 NEW: Track weakness
                update_weakness(topic, score)

                total_score += score
                attempted += 1

                print("\nEvaluation:")
                print(evaluation)
                print()

            max_score = attempted * 10 if attempted else 0
            percent = (total_score / max_score * 100) if max_score else 0.0

            print("Session Summary:")
            print(f"Topic: {topic}")
            print(f"Questions attempted: {attempted}")
            print(f"Total score: {total_score}/{max_score}")
            print(f"Overall percentage: {percent:.1f}%")

            # 🔥 NEW: Show weakness report
            print("\n📊 Performance Report:")
            print(generate_report())

        print()


if __name__ == "__main__":
    main()