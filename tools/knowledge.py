from typing import Any, cast
import numpy as np
from numpy.typing import NDArray
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient, models
from config import Settings
from schema import KnowledgeToolSearchResult
from tools.base import BaseTool, ToolResult


class KnowledgeTool(BaseTool):
    def __init__(self, settings: Settings):
        super().__init__("knowledge", "从本地知识库检索与问题相关的资料", settings)

        self._embedding_model = BGEM3FlagModel(
            settings.embedding_model_name,
            use_fp16=True,
        )

        self._reranker = CrossEncoder(
            settings.reranker_model_name,
            max_length=512,
        )

        self._qdrant_client = QdrantClient(
          url=settings.vector_db_base_url
        )

    def _search(self, query: str) -> list[KnowledgeToolSearchResult]:
        output = self._embedding_model.encode(
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

        results = self._qdrant_client.query_points(
            collection_name=self._settings.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_query.tolist(),
                    using=self._settings.dense_vector_name,
                    limit=self._settings.dense_top_k,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using=self._settings.sparse_vector_name,
                    limit=self._settings.sparse_top_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=self._settings.fusion_top_k,
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

        rerank_scores = self._reranker.predict(pairs)

        ranked_results = sorted(
            zip(candidates, rerank_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        search_results: list[KnowledgeToolSearchResult] = []
        for point, rerank_score in ranked_results:
            payload = point.payload or {}

            search_results.append(
                KnowledgeToolSearchResult(
                    chunk_id=str(point.id),
                    text=str(payload["text"]),
                    source=str(payload["source"]),
                    chunk_index=int(payload["chunk_index"]),
                    metadata=payload.get("metadata", {}),
                    fusion_score=float(point.score),
                    rerank_score=float(rerank_score),
                )
            )

        return search_results

    def run(self, **kwargs: object) -> ToolResult:
        query = str(kwargs.get("query", "")).strip()

        if not query:
            return ToolResult(success=False, data="检索问题不能为空")

        return ToolResult(
            success=True,
            data=self._search(query),
        )
