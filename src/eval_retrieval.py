import json

from .config import EVAL_QUESTIONS_PATH
from .rag_chain import get_retriever


def load_eval_questions(path=EVAL_QUESTIONS_PATH) -> list[dict]:
    if not path.exists():
        return []

    questions = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {line}") from exc

        if "question" not in item or "expected_source" not in item:
            raise ValueError(
                f"Line {line_number} must contain question and expected_source fields."
            )
        questions.append(item)

    return questions


def source_matches(doc, expected_source: str) -> bool:
    metadata = doc.metadata
    candidates = {
        metadata.get("source", ""),
        metadata.get("file_name", ""),
    }
    return any(expected_source == value or expected_source in value for value in candidates)


def evaluate() -> None:
    questions = load_eval_questions()
    if not questions:
        print("No eval questions found in eval/questions.jsonl.")
        return

    retriever = get_retriever()
    hits = 0

    for item in questions:
        question = item["question"]
        expected_source = item["expected_source"]
        docs = retriever.invoke(question)
        hit = any(source_matches(doc, expected_source) for doc in docs)

        if hit:
            hits += 1
            print(f"PASS {question}")
        else:
            returned_sources = sorted(
                {
                    doc.metadata.get("file_name") or doc.metadata.get("source", "unknown")
                    for doc in docs
                }
            )
            print(f"FAIL {question}")
            print(f"  expected: {expected_source}")
            print(f"  returned: {', '.join(returned_sources) if returned_sources else 'none'}")

    total = len(questions)
    rate = hits / total * 100
    print(f"retrieval hit rate: {hits}/{total} = {rate:.0f}%")


if __name__ == "__main__":
    evaluate()
