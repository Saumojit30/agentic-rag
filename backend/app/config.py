"""App configuration loaded from environment variables."""

import os
from pathlib import Path


class Settings:
    # --- LLM (any OpenAI-compatible API: Groq, OpenAI, Together, Ollama...) ---
    api_key: str = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v1_5")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt OS 120 B")

    # --- Retrieval & Agent ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "600"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1536"))
    max_agent_iterations: int = int(os.getenv("MAX_AGENT_ITERATIONS", "6"))
    
    # --- Hybrid Search & Reranking ---
    hybrid_alpha: float = float(os.getenv("HYBRID_ALPHA", "0.5"))
    reranker_provider: str = os.getenv("RERANKER_PROVIDER", "none")  # local, cohere, none
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "4"))          # final top documents for LLM context
    hybrid_top_k: int = int(os.getenv("HYBRID_TOP_K", "15"))         # initial search depth for reranking candidates

    # --- Storage & Caching ---
    db_path: str = os.getenv("DB_PATH", str(Path(__file__).parent.parent.parent / "data" / "financial_intelligence.db"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "86400"))

    # --- API ---
    cors_origins: list[str] = [
        o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()
    ]

    @property
    def mock_mode(self) -> bool:
        """No API key -> run the pipeline in Mock Mode."""
        return not self.api_key.strip()


settings = Settings()
