import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
CHUNK_SIZE = 800


def load_database():
    index = faiss.read_index(str(BASE_DIR / "journey_to_the_west.index"))

    with open(BASE_DIR / "chunks.json", "r", encoding="utf-8") as file:
        store = json.load(file)

    if index.ntotal != len(store["chunks"]):
        raise ValueError("FAISS 索引与 chunks.json 数量不一致")

    model = SentenceTransformer(store["embedding_model"])
    return model, index, store["chunks"]


def search(question, model, index, chunks, top_k=5):
    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
    )
    query_vector = np.asarray(query_vector, dtype="float32")

    scores, ids = index.search(query_vector, top_k)

    results = []
    for score, chunk_id in zip(scores[0], ids[0]):
        if chunk_id == -1:
            continue

        results.append(
            {
                "id": int(chunk_id),
                "score": float(score),
                "text": chunks[chunk_id],
            }
        )

    return results


if __name__ == "__main__":
    model, index, chunks = load_database()
    results = search(
        question="大闹天宫",
        model=model,
        index=index,
        chunks=chunks,
    )
    print(results)
