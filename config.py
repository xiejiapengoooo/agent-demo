from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
    )

    openai_base_url: str = ""
    openai_model: str = ""
    openai_api_key: SecretStr | None = None

    vector_db_base_url: str = ""
    collection_name: str = "demo_rag"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    dense_top_k: int = 30
    sparse_top_k: int = 30
    fusion_top_k: int = 30
    final_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
