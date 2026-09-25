"""
Centralized configuration using pydantic-settings.

All hyperparameters, paths, and API keys live here.
Override any value by setting the corresponding environment variable
or by creating a .env file in the project root.

Example:
    export OLLAMA_MODEL=mistral:7b
    export TOP_K=3
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables or .env file.

    All fields have sensible defaults so the project runs out of the box
    without any configuration. Override only what you need to change.
    """

    # --- LLM ---
    ollama_model: str = Field(
        default="qwen2.5:7b",
        description="Ollama model name used for answer generation."
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama server."
    )

    # --- Embeddings ---
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="SentenceTransformers model name for chunk and query embedding."
    )
    embedding_dim: int = Field(
        default=384,
        description="Dimensionality of the embedding vectors (384 for bge-small-en-v1.5)."
    )

    # --- Chunking ---
    chunk_size: int = Field(
        default=2000,
        description="Max characters per chunk (~500 tokens at 4 chars/token)."
    )
    chunk_overlap: int = Field(
        default=200,
        description="Characters of overlap between consecutive chunks."
    )

    # --- Retrieval ---
    top_k: int = Field(
        default=5,
        description="Number of chunks to retrieve per query."
    )

    # --- LanceDB ---
    lancedb_path: str = Field(
        default="./data/lancedb",
        description="Local filesystem path where LanceDB stores its data."
    )
    lancedb_table: str = Field(
        default="chunks",
        description="Name of the LanceDB table that stores chunk vectors."
    )

    # --- Pinecone ---
    pinecone_api_key: str = Field(
        default="",
        description="Pinecone API key. Set via environment variable or .env file."
    )
    pinecone_index_name: str = Field(
        default="fin-rag",
        description="Name of the Pinecone index to read from and write to."
    )
    pinecone_cloud: str = Field(
        default="aws",
        description="Cloud provider for the serverless Pinecone index (e.g. 'aws', 'gcp')."
    )
    pinecone_region: str = Field(
        default="us-east-1",
        description="Region for the serverless Pinecone index (e.g. 'us-east-1')."
    )
    # Legacy alias kept for backwards compatibility — not used by the serverless SDK
    pinecone_environment: str = Field(
        default="us-east-1-aws",
        description="[Legacy] Pinecone pod environment. Use pinecone_cloud + pinecone_region instead."
    )

    # --- Logging ---
    log_file: str = Field(
        default="./logs/query_log.jsonl",
        description="Path to the JSON Lines file where per-query logs are written."
    )

    # --- Eval ---
    golden_set_path: str = Field(
        default="./eval/golden_set.json",
        description="Path to the hand-written golden Q&A evaluation set."
    )
    eval_report_path: str = Field(
        default="./eval/eval_report.json",
        description="Path where the eval harness writes its output report."
    )
    eval_judge_model: str = Field(
        default="gpt-4o-mini",
        description="Model used for LLM-as-judge eval scoring. Uses OpenAI API."
    )

    # --- OpenAI (used for eval judge) ---
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for LLM judge scoring. Set via .env file."
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Module-level singleton — import this anywhere in the project.
# Example: from config import settings; print(settings.top_k)
settings = Settings()

