from app.router import route_query
from app.agents.tutor import tutor_agent_stream
from app.agents.examiner import examiner_agent

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
            pending = ""

            for piece in tutor_agent_stream(query):
                pending += piece
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    print(line)

            if pending:
                print(pending)

        # 🧪 Examiner Agent (Normal Output)
        elif route == "examiner":
            response = examiner_agent(query)
            print(response)

        print()


if __name__ == "__main__":
    main()