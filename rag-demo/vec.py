from pathlib import Path

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from langchain_community.vectorstores import FAISS
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "source"
MINERU_OUTPUT_DIR = BASE_DIR / "mineru-output"
DB_DIR = BASE_DIR / "db"
EMBED_MODEL_ID = "BAAI/bge-m3"

source_files = list(SOURCE_DIR.rglob("*.pdf"))
mineru_output_files = list(MINERU_OUTPUT_DIR.rglob("*_content_list_v2.json"))

tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
)


def get_page_no(metadata: dict) -> int | None:
    dl_meta = metadata.get("dl_meta", {})
    doc_items = dl_meta.get("doc_items", [])

    for item in doc_items:
        for prov in item.get("prov", []):
            page_no = prov.get("page_no")
            if page_no is not None:
                return page_no

    return None


if __name__ == "__main__":
    all_docs = []

    for source_file in source_files:
        loader = DoclingLoader(
            file_path=str(source_file),
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(
                tokenizer=tokenizer,
                merge_peers=True,
            ),
        )

        docs = loader.load()
        for index, doc in enumerate(docs):
            doc.metadata.update(
                {
                    "document_id": source_file.stem,
                    "chunk_id": f"{source_file.stem}-{index}",
                    "source": str(source_file),
                    "page_no": get_page_no(doc.metadata),
                }
            )

        all_docs.extend(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    vector_store = FAISS.from_documents(
        documents=all_docs,
        embedding=embeddings,
    )

    vector_store.save_local(str(DB_DIR))

    print(f"chunks: {len(all_docs)}")
    print(f"index saved to: {DB_DIR}")
