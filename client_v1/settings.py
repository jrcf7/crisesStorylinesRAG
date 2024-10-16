from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class EmmRetrieversSettings(BaseSettings):
    API_BASE: str
    API_KEY: SecretStr

    OPENAI_API_BASE_URL: str
    OPENAI_API_KEY: SecretStr

    LANGCHAIN_API_KEY: SecretStr

    DEFAULT_CLUSTER: str = "rag-os"
    DEFAULT_INDEX: str = "mine_e_emb-rag_live"

    DEFAULT_TIMEOUT: int = 120

    model_config = SettingsConfigDict(env_prefix="EMM_RETRIEVERS_", env_file="../.env")



