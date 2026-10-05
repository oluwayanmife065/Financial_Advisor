"""
Centralized configuration using pydantic-settings.

All hyperparameters, paths, and API keys live here.
Override any value by setting the corresponding environment variable
or by creating a .env file in the project root.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables or .env file.
    """

    # --- LLM Backend ---
    llm_backend: str = Field(
        default="groq",
        description="LLM backend: 'groq' (cloud, hosted default) or 'ollama' (local dev)."
    )
    allow_ollama: bool = Field(
        default=False,
        description="Enable local Ollama backend toggle. Keep False on cloud deployment."
    )

    # --- Groq (cloud LLM — default for hosted/Streamlit Cloud) ---
    groq_api_key: str = Field(
        default="",
        description="Groq API key. Get one at console.groq.com."
    )
    groq_model: str = Field(
        default="qwen/qwen3.8-27b",
        description="Groq model name."
    )

    # --- Ollama (local LLM — optional, local dev only) ---
    ollama_model: str = Field(
        default="qwen2.5:7b",
        description="Ollama model name used for answer generation (local dev only)."
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama server (local dev only)."
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
        description="Cloud provider for the serverless Pinecone index."
    )
    pinecone_region: str = Field(
        default="us-east-1",
        description="Region for the serverless Pinecone index."
    )
    pinecone_environment: str = Field(
        default="us-east-1-aws",
        description="[Legacy] Pinecone pod environment."
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
        description="Model used for LLM-as-judge eval scoring."
    )

    # --- OpenAI (used for eval judge) ---
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for LLM judge scoring."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
