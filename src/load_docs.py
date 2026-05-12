from pathlib import Path

from langchain_core.documents import Document

from .config import RAW_DATA_DIR


SUPPORTED_EXTENSIONS = {".md", ".txt"}


def iter_source_files(data_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """Return supported local document files under data_dir."""
    if not data_dir.exists():
        return []

    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_documents(data_dir: Path = RAW_DATA_DIR) -> list[Document]:
    """Load .md and .txt files as LangChain Documents with source metadata."""
    documents: list[Document] = []

    for path in iter_source_files(data_dir):
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue

        relative_source = path.relative_to(data_dir).as_posix()
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": relative_source,
                    "file_name": path.name,
                },
            )
        )

    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"loaded documents: {len(docs)}")
    for doc in docs:
        print(f"- {doc.metadata['source']}")
