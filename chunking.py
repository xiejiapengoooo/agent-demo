from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
from FlagEmbedding import BGEM3FlagModel


converter = DocumentConverter()

result = converter.convert("谢佳鹏.pdf")

document = result.document

chunker = HybridChunker()

chunks = list(chunker.chunk(document))

final_chunks: list[dict[str, Any]] = []

for i, chunk in enumerate(chunks):
    final_chunks.append(
        {
            "id": f"pdf_{i:06d}",
            "text": chunk.text,
            "metadata": {
                "source": "谢佳鹏.pdf",
                "chunk_index": i,
                "docling_meta": chunk.meta,
            },
        }
    )

texts = [chunk["text"] for chunk in final_chunks]

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True,
)

output = model.encode(
    texts,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False,
)

client = QdrantClient(url="http://localhost:6333")

collection_name = "pdf_rag"

if not client.collection_exists(collection_name):
    _ = client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )

points: list[PointStruct] = []

dense_vecs = cast(NDArray[np.float32], output["dense_vecs"])
sparse_vectors = cast(list[dict[str, float]], output["lexical_weights"])

for i, chunk in enumerate(chunks):
    dense = dense_vecs[i]
    sparse = sparse_vectors[i]

    sparse_vector = models.SparseVector(
        indices=[int(k) for k in sparse],
        values=[float(v) for v in sparse.values()],
    )

    point = models.PointStruct(
        id=i,
        vector={
            "dense": dense.tolist(),
            "sparse": sparse_vector,
        },
        payload={
            "text": chunk.text,
            "source": "谢佳鹏.pdf",
            "chunk_index": i,
            "metadata": chunk.meta,
        },
    )

    points.append(point)

_ = client.upsert(collection_name="pdf_rag", points=points)

print(client.count(collection_name="pdf_rag"))
