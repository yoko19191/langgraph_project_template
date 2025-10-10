"""Document compressor initialization utilities.

This module provides a unified interface for initializing document compressors
(rerankers) from various providers, following the same design pattern as init_chat_model.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import BaseDocumentCompressor

from src.app.common.utils.load import _get_api_key, _get_env_var

logger = logging.getLogger(__name__)


@dataclass
class RerankProviderConfig:
    """Rerank model provider configuration."""

    package: str  # Required package name
    module: str  # Module to import
    class_name: str  # Model class name
    api_key_env: str | None = None  # API key environment variable
    base_url_env: str | None = None  # Base URL environment variable
    default_base_url: str | None = None  # Default base URL when env var is not set
    default_model: str | None = None  # Default model name
    auto_load_api_key: bool = True  # Whether class auto-loads API key from env


# Configuration-driven provider registry
RERANK_PROVIDERS: dict[str, RerankProviderConfig] = {
    "siliconflow": RerankProviderConfig(
        package="custom",  # Custom implementation
        module="src.app.common.models.siliconflow",
        class_name="SiliconflowRerank",
        api_key_env="SILICONFLOW_API_KEY",
        base_url_env="SILICONFLOW_BASE_URL",
        default_base_url="https://api.siliconflow.cn/v1",
        default_model="BAAI/bge-reranker-v2-m3",
        auto_load_api_key=False,  # We manage API key manually
    ),
    "dmxapi": RerankProviderConfig(
        package="custom",  # Custom implementation
        module="src.app.common.models.dmxapi",
        class_name="DMXAPIRerank",
        api_key_env="DMXAPI_API_KEY",
        base_url_env="DMXAPI_BASE_URL",
        default_base_url="https://www.dmxapi.cn/v1",
        default_model="qwen3-reranker-8b",
        auto_load_api_key=False,  # We manage API key manually
    ),
    "xinference": RerankProviderConfig(
        package="langchain-xinference",
        module="langchain_xinference",
        class_name="XinferenceRerank",
        default_model="qwen3-reranker-0.6b",
        auto_load_api_key=True,  # LangChain package auto-loads
    ),
    "dashscope": RerankProviderConfig(
        package="langchain-community",
        module="langchain_community.document_compressors",
        class_name="DashScopeRerank",
        api_key_env="DASHSCOPE_API_KEY",
        default_model="gte-rerank",
        auto_load_api_key=True,  # LangChain package auto-loads from env
    ),
    "cohere": RerankProviderConfig(
        package="langchain-cohere",
        module="langchain_cohere",
        class_name="CohereRerank",
        api_key_env="COHERE_API_KEY",
        default_model="rerank-english-v3.0",
        auto_load_api_key=True,  # LangChain package auto-loads
    ),
    "jina": RerankProviderConfig(
        package="langchain-community",
        module="langchain_community.document_compressors",
        class_name="JinaRerank",
        api_key_env="JINA_API_KEY",
        default_model="jina-reranker-v1-base-en",
        auto_load_api_key=True,  # LangChain package auto-loads
    ),
}


def init_rerank_model(
    model: str | None = None,
    *,
    model_provider: str | None = None,
    **kwargs: Any,
) -> BaseDocumentCompressor:
    """Initialize a document compressor (reranker) from various providers.

    This function provides a unified interface for creating reranker instances
    across different providers, following the same design pattern as init_chat_model.

    Note: Unlike init_chat_model, this function requires explicit model_provider specification.
    No automatic provider inference is performed.

    Supported providers:
    - siliconflow: SiliconFlow rerank API (supports instruction, max_chunks_per_doc, overlap_tokens)
    - dmxapi: DMXAPI rerank API (basic parameters only)
    - xinference: Xinference rerank models
    - dashscope: Alibaba Dashscope rerank
    - cohere: Cohere rerank API
    - jina: Jina AI rerank

    Args:
        model: Model name/identifier. If None, uses provider's default model.
        model_provider: Provider name (required). Must be one of the supported providers.
        **kwargs: Additional parameters passed to the specific reranker class.
            Common parameters:
            - top_n: Number of documents to return
            - api_key: Provider API key (overrides environment variable)

            SiliconFlow-specific:
            - instruction: Optional reranking guidance
            - max_chunks_per_doc: Max chunks for long documents
            - overlap_tokens: Token overlap between chunks

            Xinference-specific:
            - model_uid: Model unique identifier
            - server_url: Xinference server URL

    Returns:
        BaseDocumentCompressor: Initialized reranker instance.

    Raises:
        ValueError: If model_provider is not specified or unsupported.
        ImportError: If required provider package is not installed.

    Examples:
        >>> # SiliconFlow with instruction
        >>> reranker = init_rerank_model(
        ...     model="BAAI/bge-reranker-v2-m3",
        ...     model_provider="siliconflow",
        ...     instruction="Focus on technical relevance",
        ...     top_n=5
        ... )

        >>> # DMXAPI basic usage
        >>> reranker = init_rerank_model(
        ...     model="qwen3-reranker-8b",
        ...     model_provider="dmxapi",
        ...     top_n=3
        ... )

        >>> # Xinference
        >>> reranker = init_rerank_model(
        ...     model_provider="xinference",
        ...     model_uid="qwen3-reranker-0.6b",
        ...     server_url="http://localhost:9997"
        ... )
    """
    if not model_provider:
        raise ValueError(
            "model_provider is required. Please specify one of: "
            f"{', '.join(RERANK_PROVIDERS.keys())}"
        )

    # Normalize provider name
    normalized_provider = _normalize_provider_name(model_provider)

    # Check if provider is supported
    if normalized_provider not in RERANK_PROVIDERS:
        raise ValueError(
            f"Unsupported model_provider: {model_provider}. "
            f"Supported providers: {', '.join(RERANK_PROVIDERS.keys())}"
        )

    # Create model using factory
    return _create_rerank_model(
        model=model,
        model_provider=normalized_provider,
        **kwargs,
    )


def _normalize_provider_name(provider: str) -> str:
    """Normalize provider name and handle aliases.

    Args:
        provider: Provider name to normalize.

    Returns:
        Normalized provider name.
    """
    # Alias mapping
    alias_map = {
        "qwen": "dashscope",
        "sf": "siliconflow",
    }

    normalized = provider.lower().strip().replace("-", "_")
    return alias_map.get(normalized, normalized)


def _create_rerank_model(
    model: str | None,
    model_provider: str,
    **kwargs: Any,
) -> BaseDocumentCompressor:
    """Unified rerank model creation factory function.

    Args:
        model: Model name/identifier.
        model_provider: Normalized provider name.
        **kwargs: Additional model parameters.

    Returns:
        Initialized reranker instance.

    Raises:
        ImportError: If required module cannot be imported.
        AttributeError: If model class not found in module.
    """
    config = RERANK_PROVIDERS[model_provider]

    # Dynamic import
    try:
        module = importlib.import_module(config.module)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module {config.module}. "
            f"Please install {config.package}: pip install {config.package}"
        ) from e

    # Get model class
    try:
        model_class = getattr(module, config.class_name)
    except AttributeError as e:
        raise AttributeError(
            f"Class {config.class_name} not found in module {config.module}"
        ) from e

    # Build parameters
    params = _build_rerank_params(config, model, **kwargs)

    # Create and return instance
    return model_class(**params)


def _build_rerank_params(
    config: RerankProviderConfig, model: str | None, **kwargs: Any
) -> dict[str, Any]:
    """Build reranker parameters dictionary.

    Args:
        config: Provider configuration.
        model: Model name/identifier.
        **kwargs: Additional parameters.

    Returns:
        Complete parameters dictionary for model initialization.
    """
    params = {**kwargs}

    # Add model name
    if model:
        # Xinference uses model_uid instead of model
        if config.class_name == "XinferenceRerank":
            params["model_uid"] = model
        else:
            params["model"] = model
    elif config.default_model:
        # Use default model if not specified
        if config.class_name == "XinferenceRerank":
            params.setdefault("model_uid", config.default_model)
        else:
            params.setdefault("model", config.default_model)

    # Add API key (only for custom implementations that don't auto-load)
    if not config.auto_load_api_key and config.api_key_env and "api_key" not in params:
        try:
            api_key = _get_api_key(config.api_key_env)
            # Extract string value from SecretStr if present
            if api_key is not None:
                params["api_key"] = (
                    api_key.get_secret_value()
                    if hasattr(api_key, "get_secret_value")
                    else api_key
                )
        except ValueError:
            # API key not required for some providers (e.g., xinference)
            logger.debug(
                f"API key {config.api_key_env} not found, continuing without it"
            )

    # Add base URL
    if config.base_url_env:
        base_url = _get_env_var(config.base_url_env)
        if base_url:  # Use environment variable if set
            params.setdefault("base_url", base_url)
        elif config.default_base_url:  # Use default if env var not set
            params.setdefault("base_url", config.default_base_url)

    # Handle Xinference server_url
    if config.class_name == "XinferenceRerank" and "server_url" not in params:
        server_url = _get_env_var("XINFERENCE_SERVER_URL")
        if server_url:
            params["server_url"] = server_url
        else:
            params["server_url"] = "http://localhost:9997"

    return params
