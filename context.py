import re
import unicodedata
from collections.abc import Callable
from typing import Any
from schema import (
    Citation,
    ContextChunk,
    PackedContext,
    SearchResult,
)


CONTEXT_SEPARATOR = "\n\n---\n\n"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def are_adjacent(
    left: SearchResult,
    right: SearchResult,
) -> bool:
    return (
        left.source == right.source
        and extract_section(left.metadata) == extract_section(right.metadata)
        and right.chunk_index == left.chunk_index + 1
    )


def find_text_overlap(
    left: str,
    right: str,
    min_overlap_chars: int = 20,
) -> int:
    left = left.rstrip()
    right = right.lstrip()
    max_overlap = min(len(left), len(right))

    for overlap_size in range(
        max_overlap,
        min_overlap_chars - 1,
        -1,
    ):
        if left[-overlap_size:] == right[:overlap_size]:
            return overlap_size

    return 0


def merge_text(left: str, right: str) -> str | None:
    left = left.rstrip()
    right = right.lstrip()
    overlap_size = find_text_overlap(left, right)

    if overlap_size == 0:
        return None

    return left + right[overlap_size:]


def extract_section(metadata: dict[str, Any]) -> str:
    headings = metadata.get("headings")

    if not isinstance(headings, list):
        return ""

    valid_headings = [
        str(heading).strip() for heading in headings if str(heading).strip()
    ]

    return " > ".join(valid_headings)


def extract_pages(metadata: dict[str, Any]) -> list[int]:
    pages: set[int] = set()
    doc_items = metadata.get("doc_items")

    if not isinstance(doc_items, list):
        return []

    for doc_item in doc_items:
        if not isinstance(doc_item, dict):
            continue

        provisions = doc_item.get("prov")

        if not isinstance(provisions, list):
            continue

        for provision in provisions:
            if not isinstance(provision, dict):
                continue

            page_no = provision.get("page_no")

            if isinstance(page_no, int):
                pages.add(page_no)

    return sorted(pages)


def to_context_chunk(result: SearchResult) -> ContextChunk:
    return ContextChunk(
        text=result.text,
        source=result.source,
        section=extract_section(result.metadata),
        pages=extract_pages(result.metadata),
        chunk_ids=[result.chunk_id],
        chunk_indexes=[result.chunk_index],
        rerank_score=result.rerank_score,
    )


def merge_adjacent_results(
    results: list[SearchResult],
) -> list[ContextChunk]:
    if not results:
        return []

    # 检查相邻关系时必须按文档顺序排列。
    document_order = sorted(
        results,
        key=lambda result: (
            result.source,
            extract_section(result.metadata),
            result.chunk_index,
        ),
    )

    merged_results: list[ContextChunk] = []
    previous_result = document_order[0]
    current = to_context_chunk(previous_result)

    for result in document_order[1:]:
        merged_text = None

        if are_adjacent(previous_result, result):
            merged_text = merge_text(current.text, result.text)

        if merged_text is None:
            merged_results.append(current)
            current = to_context_chunk(result)
        else:
            current.text = merged_text
            current.pages = sorted(set(current.pages + extract_pages(result.metadata)))
            current.chunk_ids.append(result.chunk_id)
            current.chunk_indexes.append(result.chunk_index)
            current.rerank_score = max(
                current.rerank_score,
                result.rerank_score,
            )

        previous_result = result

    merged_results.append(current)

    return sorted(
        merged_results,
        key=lambda item: item.rerank_score,
        reverse=True,
    )


# 去掉文案完全相同，保留最高分数的结果
def deduplicate_exact_text(
    results: list[SearchResult],
) -> list[SearchResult]:
    sorted_results = sorted(
        results,
        key=lambda result: result.rerank_score,
        reverse=True,
    )

    unique_results: list[SearchResult] = []
    seen_chunk_ids: set[str] = set()
    seen_texts: set[str] = set()

    for result in sorted_results:
        normalized_text = normalize_text(result.text)

        if result.chunk_id in seen_chunk_ids:
            continue

        if normalized_text in seen_texts:
            continue

        seen_chunk_ids.add(result.chunk_id)
        seen_texts.add(normalized_text)
        unique_results.append(result)

    return unique_results


def prepare_context_chunks(
    results: list[SearchResult],
) -> list[ContextChunk]:
    unique_results = deduplicate_exact_text(results)
    merged_results = merge_adjacent_results(unique_results)

    return merged_results


def format_pages(pages: list[int]) -> str:
    if not pages:
        return "未知"

    return "、".join(str(page) for page in pages)


def format_context_block(
    citation_id: str,
    chunk: ContextChunk,
) -> str:
    section = chunk.section or "未知"

    return (
        f"[{citation_id}]\n"
        f"文档：{chunk.source}\n"
        f"章节：{section}\n"
        f"页码：{format_pages(chunk.pages)}\n"
        f"内容：\n{chunk.text.strip()}"
    )


def pack_context(
    chunks: list[ContextChunk],
    count_tokens: Callable[[str], int],
    max_context_tokens: int,
    max_chunks: int = 5,
) -> PackedContext:
    if max_context_tokens <= 0:
        raise ValueError("max_context_tokens 必须大于 0")

    if max_chunks <= 0:
        raise ValueError("max_chunks 必须大于 0")

    selected_chunks: list[ContextChunk] = []
    context_blocks: list[str] = []
    citations: dict[str, Citation] = {}
    token_count = 0

    sorted_chunks = sorted(
        chunks,
        key=lambda chunk: chunk.rerank_score,
        reverse=True,
    )

    for chunk in sorted_chunks:
        citation_id = f"S{len(selected_chunks) + 1}"
        block = format_context_block(citation_id, chunk)
        candidate_text = CONTEXT_SEPARATOR.join([*context_blocks, block])
        candidate_token_count = count_tokens(candidate_text)

        if candidate_token_count > max_context_tokens:
            continue

        selected_chunks.append(chunk)
        context_blocks.append(block)
        token_count = candidate_token_count
        citations[citation_id] = Citation(
            source=chunk.source,
            section=chunk.section,
            pages=chunk.pages,
            chunk_ids=chunk.chunk_ids,
        )

        if len(selected_chunks) >= max_chunks:
            break

    return PackedContext(
        text=CONTEXT_SEPARATOR.join(context_blocks),
        token_count=token_count,
        chunks=selected_chunks,
        citations=citations,
    )
