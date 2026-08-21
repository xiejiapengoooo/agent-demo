from typing import Any
from pydantic import BaseModel, Field
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextBudget:
    fixed_input_tokens: int
    max_context_tokens: int
    max_completion_tokens: int
    safety_margin_tokens: int


class KnowledgeToolSearchResult(BaseModel):
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    fusion_score: float
    rerank_score: float


class ContextChunk(BaseModel):
    text: str
    source: str
    section: str
    pages: list[int] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    chunk_indexes: list[int] = Field(default_factory=list)
    rerank_score: float


class Citation(BaseModel):
    source: str
    section: str
    pages: list[int] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)


class PackedContext(BaseModel):
    text: str
    chunks: list[ContextChunk] = Field(default_factory=list)
    citations: dict[str, Citation] = Field(default_factory=dict)
