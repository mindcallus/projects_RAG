import shutil

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)
from .load_docs import load_documents


def reset_chroma_dir() -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    for child in CHROMA_DIR.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks


def rebuild_index() -> None:
    reset_chroma_dir()

    documents = load_documents()
    if not documents:
        print("No .md or .txt documents found in data/raw/.")
        print("Add your project documents, then run: python -m src.build_index")
        return

    chunks = split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"documents: {len(documents)}")
    print(f"chunks: {len(chunks)}")
    print(f"chroma persisted to: {CHROMA_DIR}")


if __name__ == "__main__":
    rebuild_index()
