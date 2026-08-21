from langchain_core.messages import BaseMessage

from config import get_settings
from context import prepare_context_chunks, pack_context
from context_budget import ContextBudgetManager
from generation import ANSWER_PROMPT, create_llm, generate_answer
from search import search


settings = get_settings()


def answer_query(
    query: str,
    history: list[BaseMessage] | None = None,
) -> str:
    query = query.strip()
    if not query:
        return "根据现有资料无法确定。"

    llm = create_llm()
    manager = ContextBudgetManager(
        llm=llm,
        prompt=ANSWER_PROMPT,
        context_window=settings.openai_context_window,
        max_completion_tokens=settings.openai_max_completion_tokens,
        safety_margin_tokens=settings.openai_token_safety_margin,
    )
    context_counter = manager.create_context_counter(
        query=query,
        history=history,
    )
    results = search(query)
    chunks = prepare_context_chunks(results)
    packed = pack_context(
        chunks=chunks,
        context_counter=context_counter,
        max_chunks=5,
    )
    _ = manager.validate(
        query=query,
        context=packed.text,
        history=history,
    )
    return generate_answer(
        query=query,
        context=packed,
        llm=llm,
        history=history,
    )


def run() -> None:
    answer = answer_query("我是谁？")
    print(answer)


if __name__ == "__main__":
    run()
