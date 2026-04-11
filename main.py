from app.router import route_query
from app.agents.tutor import tutor_agent_stream, tutor_agent, is_diagram_query
from app.agents.examiner import examiner_agent
from app.agents.evaluator import evaluate_answer, extract_score
from app.memory.weakness_tracker import update_weakness, generate_report
from app.utils.mermaid_parser import extract_mermaid_block


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


def main():
    while True:
        query = input("Ask (type 'exit' to quit): ").strip()

        if not query:
            continue

        if query.lower() == "exit":
            print("Goodbye 👋")
            break

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