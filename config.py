from pathlib import Path
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
    )

    docs_dir: Path = Path("docs")

    openai_base_url: str = ""
    openai_model: str = ""
    openai_api_key: SecretStr | None = None

    vector_db_base_url: str = ""
    embedding_model_name: str = "BAAI/bge-m3"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    collection_name: str = "demo_rag"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    dense_top_k: int = 30
    sparse_top_k: int = 30
    fusion_top_k: int = 30
    embedding_size: int = 1024
    qdrant_upload_batch_size: int = 64


@lru_cache
def get_settings() -> Settings:
    return Settings()
