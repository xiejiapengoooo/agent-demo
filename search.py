from typing import Any, cast
import numpy as np
from numpy.typing import NDArray
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient, models


QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "pdf_rag"

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

DENSE_TOP_K = 30
SPARSE_TOP_K = 30

FUSION_TOP_K = 30

FINAL_TOP_K = 5


print("Loading BGE-M3...")
embedding_model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True,
)

print("Loading Reranker...")
reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    max_length=512,
)

client = QdrantClient(url=QDRANT_URL)


def search(query: str):
    output = embedding_model.encode(
        [query],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense_vecs = cast(NDArray[np.float32], output["dense_vecs"])
    sparse_vectors = cast(list[dict[str, float]], output["lexical_weights"])

    dense_query = dense_vecs[0]
    sparse_query = sparse_vectors[0]

    sparse_vector = models.SparseVector(
        indices=[int(k) for k in sparse_query],
        values=[float(v) for v in sparse_query.values()],
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_query.tolist(),
                using=DENSE_VECTOR_NAME,
                limit=DENSE_TOP_K,
            ),
            models.Prefetch(
                query=sparse_vector,
                using=SPARSE_VECTOR_NAME,
                limit=SPARSE_TOP_K,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=FUSION_TOP_K,
        with_payload=True,
    )

    candidates = []
    pairs = []

    for point in results.points:
        if not point.payload:
            continue

        text = point.payload.get("text", "")

        if not text:
            continue

        candidates.append(point)
        pairs.append([query, text])

    if not pairs:
        return []

    rerank_scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(candidates, rerank_scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    return ranked_results[:FINAL_TOP_K]
