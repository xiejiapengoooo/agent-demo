from generation import generate_answer
from search import search
from context import prepare_context_chunks, pack_context


def run():
    query = "我是谁？".strip()
    results = search(query)
    chunks = prepare_context_chunks(results)
    packed = pack_context(
        chunks=chunks,
        count_tokens=count_tokens,
        max_context_tokens=6000,
        max_chunks=5,
    )
    generate_answer(query, packed, llm)


if __name__ == "__main__":
    run()
