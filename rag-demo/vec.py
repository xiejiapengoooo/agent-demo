import json
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "source"
MINERU_OUTPUT_DIR = BASE_DIR / "mineru-output"
DB_DIR = BASE_DIR / "db"
EMBED_MODEL_ID = "BAAI/bge-m3"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
SKIP_TYPES = {"header", "footer", "page_header", "page_footer", "page_number"}


source_files: list[Path] = []


class HTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"br", "p", "tr"}:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def clean_text(text: str) -> str:
    lines = [" ".join(line.split()).strip(" |") for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def html_to_text(value: str) -> str:
    parser = HTMLTextParser()
    parser.feed(value)
    return clean_text("".join(parser.parts))


def run_mineru() -> list[Path]:
    source_files = sorted(SOURCE_DIR.rglob("*.pdf"))
    if not source_files:
        raise FileNotFoundError(f"在 {SOURCE_DIR} 中没有找到 PDF 文件")

    if MINERU_OUTPUT_DIR.exists():
        if not MINERU_OUTPUT_DIR.is_dir():
            raise NotADirectoryError(MINERU_OUTPUT_DIR)
        shutil.rmtree(MINERU_OUTPUT_DIR)
    MINERU_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "mineru.cli.client",
            "--path",
            str(SOURCE_DIR),
            "--output",
            str(MINERU_OUTPUT_DIR),
            "--backend",
            "pipeline",
            "--method",
            "auto",
            # "--lang",
            # "ch",
        ],
        check=True,
    )

    return source_files


def collect_text(value: Any) -> list[str]:
    parts: list[str] = []

    if isinstance(value, str):
        parts.append(value)
    elif isinstance(value, list):
        for item in value:
            parts.extend(collect_text(item))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key == "html" and isinstance(item, str):
                parts.append(html_to_text(item))
            elif isinstance(item, str):
                if key in {
                    "content",
                    "text",
                    "latex",
                    "formula",
                    "caption",
                    "ocr_text",
                } or key.endswith("_content"):
                    parts.append(item)
            elif isinstance(item, (dict, list)):
                parts.extend(collect_text(item))

    return parts


def load_blocks(path: Path) -> list[dict]:
    pages = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(pages, list):
        raise TypeError(f"expected a list in {path}")

    blocks: list[dict] = []
    for page_index, page in enumerate(pages):
        if not isinstance(page, list):
            continue

        for block in page:
            if not isinstance(block, dict):
                continue

            block_type = str(block.get("type", "unknown"))
            if block_type in SKIP_TYPES:
                continue

            text = clean_text("\n".join(collect_text(block.get("content"))))
            if text:
                blocks.append(
                    {
                        "text": text,
                        "type": block_type,
                        "page_no": page_index + 1,
                    }
                )

    return blocks


def build_documents(path: Path, splitter) -> list[Document]:
    document_id = path.name.removesuffix("_content_list_v2.json")

    try:
        source = next(
            (str(pdf) for pdf in source_files if pdf.stem == document_id),
        )
    except StopIteration:
        raise FileNotFoundError(f"找不到文档: {document_id}")

    pages: dict[int, list[dict]] = {}
    for block in load_blocks(path):
        pages.setdefault(block["page_no"], []).append(block)

    page_documents = [
        Document(
            page_content="\n\n".join(block["text"] for block in blocks),
            metadata={
                "document_id": document_id,
                "source": source,
                "page_no": page_no,
                "block_types": ",".join(
                    dict.fromkeys(block["type"] for block in blocks)
                ),
            },
        )
        for page_no, blocks in sorted(pages.items())
    ]
    documents = splitter.split_documents(page_documents)
    for index, document in enumerate(documents):
        document.metadata["chunk_id"] = f"{document_id}_{index}"

    return documents


if __name__ == "__main__":
    source_files = run_mineru()
    mineru_output_files = sorted(MINERU_OUTPUT_DIR.rglob("*_content_list_v2.json"))
    if not mineru_output_files:
        raise FileNotFoundError(
            f"MinerU 没有生成 *_content_list_v2.json：{MINERU_OUTPUT_DIR}"
        )

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    all_docs = [
        doc
        for output_file in mineru_output_files
        for doc in build_documents(output_file, splitter)
    ]
    if not all_docs:
        raise FileNotFoundError(f"MinerU 没有解析出文档内容：{MINERU_OUTPUT_DIR}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = FAISS.from_documents(
        documents=all_docs,
        embedding=embeddings,
    )

    vector_store.save_local(str(DB_DIR))

    print(f"chunks: {len(all_docs)}")
    print(f"index saved to: {DB_DIR}")
