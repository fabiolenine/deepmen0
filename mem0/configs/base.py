import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from mem0.configs.rerankers.config import RerankerConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig

# Set up the directory path
home_dir = os.path.expanduser("~")
mem0_dir = os.environ.get("MEM0_DIR") or os.path.join(home_dir, ".mem0")


class MemoryItem(BaseModel):
    id: str = Field(..., description="The unique identifier for the text data")
    memory: str = Field(
        ..., description="The memory deduced from the text data"
    )  # TODO After prompt changes from platform, update this
    hash: Optional[str] = Field(None, description="The hash of the memory")
    # The metadata value can be anything and not just string. Fix it
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the text data")
    score: Optional[float] = Field(None, description="The score associated with the text data")
    created_at: Optional[str] = Field(None, description="The timestamp when the memory was created")
    updated_at: Optional[str] = Field(None, description="The timestamp when the memory was updated")


class MemoryDynamicsConfig(BaseModel):
    """DeepMem0 v0.2: human-memory dynamics (ACT-R base-level activation).

    Every memory lives on an evolving timeline; re-encounters reinforce it and
    activation (frequency + recency in a single term) becomes a ranking signal.
    Activation is computed lazily at query time — nothing is stored or decayed
    in the background. Memories without a reinforcement history are neutral.
    """

    enabled: bool = Field(
        description="Master switch for memory dynamics (reinforcement write-back and activation in ranking).",
        default=True,
    )
    decay: float = Field(
        description="ACT-R base-level decay exponent d (0.5 is the canonical value).",
        default=0.5,
    )
    weight: float = Field(
        description="Weight of the activation boost in ranking, both at fusion and after the"
        " reranker. 0 disables the ranking term (write-back still records the timeline).",
        default=0.15,
    )
    reinforcement_window: int = Field(
        description="Seconds after a reinforcement during which further re-encounters of the"
        " same memory have no reinforcement effect (absorbs client retries; approximates the"
        " ACT-R spacing effect). 0 disables the window.",
        default=3600,
    )
    max_timestamps: int = Field(
        description="Reinforcement timestamps retained verbatim per memory; the older tail"
        " folds into the Petrov (2006) approximation so payload stays O(K).",
        default=10,
    )
    reinforce_on_search: bool = Field(
        description="Also reinforce memories returned in the final top-k of a search"
        " (async, fire-and-forget, never blocks the hot path).",
        default=False,
    )


class MemoryConfig(BaseModel):
    vector_store: VectorStoreConfig = Field(
        description="Configuration for the vector store",
        default_factory=VectorStoreConfig,
    )
    llm: LlmConfig = Field(
        description="Configuration for the language model",
        default_factory=LlmConfig,
    )
    embedder: EmbedderConfig = Field(
        description="Configuration for the embedding model",
        default_factory=EmbedderConfig,
    )
    history_db_path: str = Field(
        description="Path to the history database",
        default=os.path.join(mem0_dir, "history.db"),
    )
    reranker: Optional[RerankerConfig] = Field(
        description="Configuration for the reranker",
        default=None,
    )
    version: str = Field(
        description="The version of the API",
        default="v1.1",
    )
    custom_instructions: Optional[str] = Field(
        description="Custom instructions for fact extraction",
        default=None,
    )
    language: str = Field(
        description="DeepMem0: ISO 639-1 language of the memory corpus (e.g. 'pt')."
        " Wires through BM25 stemming/stopwords, BM25 text normalization and the"
        " extraction prompt. 'en' preserves upstream mem0 behavior.",
        default="en",
    )
    rerank_pool: int = Field(
        description="DeepMem0: minimum candidate pool handed to the reranker"
        " (effective pool = max(2*top_k, rerank_pool)). Pools beyond ~20 measured"
        " no quality gain at 3x the latency.",
        default=20,
    )
    dynamics: MemoryDynamicsConfig = Field(
        description="DeepMem0 v0.2: human-memory dynamics (ACT-R activation) settings.",
        default_factory=MemoryDynamicsConfig,
    )


class AzureConfig(BaseModel):
    """
    Configuration settings for Azure.

    Args:
        api_key (str): The API key used for authenticating with the Azure service.
        azure_deployment (str): The name of the Azure deployment.
        azure_endpoint (str): The endpoint URL for the Azure service.
        api_version (str): The version of the Azure API being used.
        default_headers (Dict[str, str]): Headers to include in requests to the Azure API.
    """

    api_key: str = Field(
        description="The API key used for authenticating with the Azure service.",
        default=None,
    )
    azure_deployment: str = Field(description="The name of the Azure deployment.", default=None)
    azure_endpoint: str = Field(description="The endpoint URL for the Azure service.", default=None)
    api_version: str = Field(description="The version of the Azure API being used.", default=None)
    default_headers: Optional[Dict[str, str]] = Field(
        description="Headers to include in requests to the Azure API.", default=None
    )
