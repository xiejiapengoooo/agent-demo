from __future__ import annotations

import argparse
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "db"
EMBED_MODEL_ID = "BAAI/bge-m3"
DEFAULT_K = 5


def load_vector_store() -> FAISS:
    index_file = DB_DIR / "index.faiss"
    metadata_file = DB_DIR / "index.pkl"

    if not index_file.is_file() or not metadata_file.is_file():
        raise FileNotFoundError(
            f"FAISS index not found in {DB_DIR}. Run vec.py first to build the index."
        )

    return FAISS.load_local(
        str(DB_DIR),
        HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_ID,
            encode_kwargs={"normalize_embeddings": True},
        ),
        allow_dangerous_deserialization=True,
    )


def search(
    vector_store: FAISS,
    query: str,
    *,
    k: int = DEFAULT_K,
) -> list[tuple[Document, float]]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    return vector_store.similarity_search_with_score(query.strip(), k=k)


def print_results(results: list[tuple[Document, float]]) -> None:
    if not results:
        print("没有找到相关内容。")
        return

    for rank, (doc, score) in enumerate(results, start=1):
        metadata = doc.metadata
        score_text = f"{score:.4f}" if score is not None else "n/a"

        print(f"\n--- Result {rank} | score: {score_text} ---")
        print(f"document_id: {metadata.get('document_id', 'unknown')}")
        print(f"page_no: {metadata.get('page_no', 'unknown')}")
        print(f"chunk_id: {metadata.get('chunk_id', 'unknown')}")
        print(f"source: {metadata.get('source', 'unknown')}")
        print(doc.page_content.strip())


if __name__ == "__main__":
    vector_store = load_vector_store()
    results = search(
        vector_store,
        "客户经理待遇",
    )
    print_results(results)
