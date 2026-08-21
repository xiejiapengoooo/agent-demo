from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from schema import ContextBudget


class ContextBudgetExceededError(ValueError):
    pass


class ContextBudgetManager:
    def __init__(
        self,
        llm: BaseChatModel,
        prompt: ChatPromptTemplate,
        context_window: int,
        max_completion_tokens: int,
        safety_margin_tokens: int,
    ) -> None:
        if context_window <= 0:
            raise ValueError("context_window 必须大于 0")

        if max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens 必须大于 0")

        if safety_margin_tokens < 0:
            raise ValueError("safety_margin_tokens 不能小于 0")

        self.llm = llm
        self.prompt = prompt
        self.context_window = context_window
        self.max_completion_tokens = max_completion_tokens
        self.safety_margin_tokens = safety_margin_tokens

    def count_prompt_tokens(
        self,
        query: str,
        context: str,
        history: list[BaseMessage] | None = None,
    ) -> int:
        messages = self.prompt.format_messages(
            history=history or [],
            query=query.strip(),
            context=context,
        )
        return self.llm.get_num_tokens_from_messages(messages)

    def calculate(
        self,
        query: str,
        history: list[BaseMessage] | None = None,
    ) -> ContextBudget:
        # 空 Context 时，计算系统提示词、历史、当前问题和消息格式开销。
        fixed_input_tokens = self.count_prompt_tokens(
            query=query,
            context="",
            history=history,
        )

        max_context_tokens = (
            self.context_window
            - fixed_input_tokens
            - self.max_completion_tokens
            - self.safety_margin_tokens
        )

        if max_context_tokens <= 0:
            raise ContextBudgetExceededError(
                "系统提示词、历史消息、当前问题和输出预留已经超过模型窗口"
            )

        return ContextBudget(
            fixed_input_tokens=fixed_input_tokens,
            max_context_tokens=max_context_tokens,
            max_completion_tokens=self.max_completion_tokens,
            safety_margin_tokens=self.safety_margin_tokens,
        )

    def create_context_counter(
        self,
        query: str,
        budget: ContextBudget,
        history: list[BaseMessage] | None = None,
    ) -> Callable[[str], int]:
        def count_context_tokens(context: str) -> int:
            total_input_tokens = self.count_prompt_tokens(
                query=query,
                context=context,
                history=history,
            )

            return max(
                total_input_tokens - budget.fixed_input_tokens,
                0,
            )

        return count_context_tokens

    def validate(
        self,
        query: str,
        context: str,
        history: list[BaseMessage] | None = None,
    ) -> int:
        input_tokens = self.count_prompt_tokens(
            query=query,
            context=context,
            history=history,
        )

        required_tokens = (
            input_tokens + self.max_completion_tokens + self.safety_margin_tokens
        )

        if required_tokens > self.context_window:
            raise ContextBudgetExceededError(
                f"请求需要 {required_tokens} tokens，"
                f"模型窗口只有 {self.context_window} tokens"
            )

        return input_tokens
