from app.agents.tutor import tutor_agent_stream

def main():
    while True:
        query = input("Ask: ").strip()
        if not query:
            continue

        print("\nAnswer:")
        pending = ""

        for piece in tutor_agent_stream(query):
            pending += piece
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                print(line)

        if pending:
            print(pending)

        print()


if __name__ == "__main__":
    main()