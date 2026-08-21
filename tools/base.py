from abc import ABC, abstractmethod
from typing import Any
from config import Settings
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    success: bool
    data: Any


class BaseTool(ABC):
    name: str
    description: str

    def __init__(self, name: str, description: str, settings: Settings):
        self.name = name
        self.description = description
        self._settings = settings

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError
