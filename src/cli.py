from .rag_chain import ask


def main() -> None:
    print("Project Interview RAG CLI")
    print("Type exit or quit to leave.")

    while True:
        question = input("\nAsk> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        try:
            result = ask(question)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        print(f"Answer> {result['answer']}")
        print("Sources:")
        if result["sources"]:
            for source in result["sources"]:
                print(f"- {source}")
        else:
            print("- 无")


if __name__ == "__main__":
    main()
