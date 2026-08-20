from pathlib import Path
from typing import Any, cast
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from numpy.typing import NDArray
from qdrant_client import QdrantClient, models
from qdrant_client.grpc.collections_pb2 import Bool
from qdrant_client.models import PointStruct
from config import get_settings


settings = get_settings()


def get_document_paths() -> list[Path]:
    return sorted(
        path
        for path in settings.docs_dir.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(settings.docs_dir).parts)
    )


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(settings.collection_name):
        return

    _ = client.create_collection(
        collection_name=settings.collection_name,
        vectors_config={
            settings.dense_vector_name: models.VectorParams(
                size=settings.embedding_size,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            settings.sparse_vector_name: models.SparseVectorParams()
        },
    )


def build_points(
    source,
    chunks,
    embedding_output,
) -> list[PointStruct]:
    dense_vectors = cast(NDArray[np.float32], embedding_output["dense_vecs"])
    sparse_vectors = cast(list[dict[str, float]], embedding_output["lexical_weights"])
    points: list[PointStruct] = []

    for chunk_index, chunk in enumerate(chunks):
        sparse = sparse_vectors[chunk_index]
        sparse_vector = models.SparseVector(
            indices=[int(index) for index in sparse],
            values=[float(value) for value in sparse.values()],
        )
        point_id = str(uuid5(NAMESPACE_URL, f"{source}:{chunk_index}"))

        points.append(
            PointStruct(
                id=point_id,
                vector={
                    settings.dense_vector_name: dense_vectors[chunk_index].tolist(),
                    settings.sparse_vector_name: sparse_vector,
                },
                payload={
                    "text": chunk.text,
                    "source": source,
                    "chunk_index": chunk_index,
                    "metadata": chunk.meta.model_dump(mode="json"),
                },
            )
        )

    return points


def ingest_document(
    path: Path,
    converter: DocumentConverter,
    chunker: HybridChunker,
    embedding_model: BGEM3FlagModel,
    client: QdrantClient,
) -> int:
    source = path.relative_to(settings.docs_dir).as_posix()
    document = converter.convert(path).document
    chunks = list(chunker.chunk(document))
    if not chunks:
        return 0

    output = embedding_model.encode(
        [chunk.text for chunk in chunks],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    points = build_points(source, chunks, output)
    client.upload_points(
        collection_name=settings.collection_name,
        points=points,
        batch_size=settings.qdrant_upload_batch_size,
        wait=True,
    )
    return len(points)


def run() -> None:
    document_paths = get_document_paths()
    converter = DocumentConverter()
    chunker = HybridChunker()
    client = QdrantClient(url=settings.vector_db_base_url)
    ensure_collection(client)
    embedding_model = BGEM3FlagModel(
        settings.embedding_model_name,
        use_fp16=True,
    )

    failed_documents: list[tuple[Path, Exception]] = []

    for document_path in document_paths:
        try:
            chunk_count = ingest_document(
                document_path,
                converter,
                chunker,
                embedding_model,
                client,
            )
            print(f"Imported {document_path.relative_to(settings.docs_dir)}: {chunk_count} chunks")
        except Exception as error:
            failed_documents.append((document_path, error))
            print(f"Failed to import {document_path.relative_to(settings.docs_dir)}: {error}")

    print(
        f"Import completed: {len(document_paths) - len(failed_documents)} documents, "
        f"{len(failed_documents)} failures"
    )
    if failed_documents:
        raise RuntimeError("Some documents could not be imported")


if __name__ == "__main__":
    run()
