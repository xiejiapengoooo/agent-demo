from tools.base import BaseTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具已注册：{tool.name}")

        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as error:
            raise ValueError(f"未知工具：{name}") from error

    def descriptions(self) -> str:
        return "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in self._tools.values()
        )
