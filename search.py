from typing import Any, cast
import numpy as np
from numpy.typing import NDArray
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient, models
from config import get_settings


settings = get_settings()


print("Loading BGE-M3...")
embedding_model = BGEM3FlagModel(
    settings.embedding_model_name,
    use_fp16=True,
)

print("Loading Reranker...")
reranker = CrossEncoder(
    settings.reranker_model_name,
    max_length=512,
)

client = QdrantClient(url=settings.vector_db_base_url)


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
        collection_name=settings.collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_query.tolist(),
                using=settings.dense_vector_name,
                limit=settings.dense_top_k,
            ),
            models.Prefetch(
                query=sparse_vector,
                using=settings.sparse_vector_name,
                limit=settings.sparse_top_k,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=settings.fusion_top_k,
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

    return ranked_results[: settings.final_top_k]
