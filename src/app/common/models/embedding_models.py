"""Embedding model initialization module.

Provides a unified interface for initializing embedding models from various providers.
Follows the same config-driven design pattern as chat_models.py for consistency.
Supports optional caching via LangChain's CacheBackedEmbeddings.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from langchain.embeddings.base import Embeddings

from src.app.common.utils import _get_api_key, _get_env_var

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingProviderConfig:
    """Embedding model provider configuration.

    Attributes:
        package: Required package name for installation
        module: Python module path to import
        class_name: Embedding model class name
        api_key_env: Environment variable name for API key (optional)
        base_url_env: Environment variable name for base URL (optional)
        default_base_url: Default base URL when env var is not set (optional)
        special_params: Special parameters dict for provider-specific config (optional)
    """

    package: str
    module: str
    class_name: str
    api_key_env: str | None = None
    base_url_env: str | None = None
    default_base_url: str | None = None
    special_params: Dict[str, Any] | None = None


# Configuration-driven provider registry - "Explicit is better than implicit"
EMBEDDING_PROVIDERS: Dict[str, EmbeddingProviderConfig] = {
    "openai": EmbeddingProviderConfig(
        package="langchain-openai",
        module="langchain_openai",
        class_name="OpenAIEmbeddings",
        api_key_env="OPENAI_API_KEY",
    ),
    "siliconflow": EmbeddingProviderConfig(
        package="langchain-siliconflow",
        module="langchain_siliconflow",
        class_name="SiliconFlowEmbeddings",
        api_key_env="SILICONFLOW_API_KEY",
        base_url_env="SILICONFLOW_BASE_URL",
        default_base_url="https://api.siliconflow.cn/v1",
    ),
    "xinference": EmbeddingProviderConfig(
        package="langchain-xinference",
        module="langchain_xinference",
        class_name="XinferenceEmbeddings",
        special_params={
            "server_url_env": "XINFERENCE_SERVER_URL",
            "model_uid_env": "XINFERENCE_EMBEDDING_MODEL_UID",
        },
    ),
    "ollama": EmbeddingProviderConfig(
        package="langchain-ollama",
        module="langchain_ollama",
        class_name="OllamaEmbeddings",
        base_url_env="OLLAMA_BASE_URL",
        default_base_url="http://localhost:11434",
    ),
    "jinaai": EmbeddingProviderConfig(
        package="langchain-community",
        module="langchain_community.embeddings",
        class_name="JinaEmbeddings",
        api_key_env="JINA_API_KEY",
    ),
    "dashscope": EmbeddingProviderConfig(
        package="langchain-community",
        module="langchain_community.embeddings",
        class_name="DashScopeEmbeddings",
        api_key_env="DASHSCOPE_API_KEY",
    ),
    "dmxapi": EmbeddingProviderConfig(
        package="langchain-openai",
        module="langchain_openai",
        class_name="OpenAIEmbeddings",
        api_key_env="DMXAPI_API_KEY",
        base_url_env="DMXAPI_BASE_URL",
        default_base_url="https://www.DMXapi.cn/v1",
    ),
    "google": EmbeddingProviderConfig(
        package="langchain-google-genai",
        module="langchain_google_genai",
        class_name="GoogleGenerativeAIEmbeddings",
        api_key_env="GOOGLE_API_KEY",
    ),
}


def init_embedding_model(
    model: str | None = None,
    *,
    model_provider: str | None = None,
    provider: str | None = None,  # Alias for model_provider for convenience
    cache: bool = False,
    cache_dir: str | Path | None = None,
    cache_namespace: str | None = None,
    cache_in_memory: bool = False,
    **kwargs: Any,
) -> Embeddings:
    """Initialize embedding model with specified provider and optional caching.

    This function provides a unified interface for creating embedding models from
    various providers. It follows a config-driven approach for consistency and
    maintainability. Supports optional caching via LangChain's CacheBackedEmbeddings.

    Supported providers:
    - openai: OpenAI embedding models(text-embedding only)
    - siliconflow: SiliconFlow embedding models
    - xinference: Xinference local deployment
    - ollama: Ollama local models
    - jinaai: Jina AI embedding models
    - dashscope: Alibaba DashScope (Qwen) embedding models(text-embedding only)
    - dmxapi: DMXAPI OpenAI-compatible embedding models(text-embedding only)
    - google: Google Generative AI embedding models

    Args:
        model: Model name/identifier
        model_provider: Provider name (must be one of supported providers)
        provider: Alias for model_provider (for convenience)
        cache: Whether to enable embedding caching (default: False)
        cache_dir: Directory for file-based caching. If None and cache=True, 
                  uses "./embeddings_cache" (default: None)
        cache_namespace: Namespace for cache keys. If None, uses model name (default: None)
        cache_in_memory: Use in-memory caching instead of file-based (default: False)
        **kwargs: Additional model-specific parameters

    Returns:
        Embeddings: Initialized embedding model instance (optionally cached)

    Raises:
        ValueError: When provider is not specified or not supported
        ImportError: When required package is not installed
        AttributeError: When model class is not found in module

    Examples:
        >>> # Initialize OpenAI embeddings without caching
        >>> embeddings = init_embedding_model(
        ...     model="text-embedding-3-small",
        ...     model_provider="openai"
        ... )

        >>> # Initialize with file-based caching
        >>> embeddings = init_embedding_model(
        ...     model="text-embedding-3-small",
        ...     provider="openai",
        ...     cache=True,
        ...     cache_dir="./my_cache"
        ... )

        >>> # Initialize with in-memory caching
        >>> embeddings = init_embedding_model(
        ...     model="text-embedding-3-small",
        ...     provider="openai",
        ...     cache=True,
        ...     cache_in_memory=True
        ... )

        >>> # Initialize DashScope embeddings with caching
        >>> embeddings = init_embedding_model(
        ...     model="text-embedding-v4",
        ...     provider="dashscope",
        ...     cache=True
        ... )
    """
    # Handle provider alias - "Explicit is better than implicit"
    if provider is not None and model_provider is None:
        model_provider = provider
    elif provider is not None and model_provider is not None:
        raise ValueError(
            "Cannot specify both 'provider' and 'model_provider'. "
            "Please use only one of them."
        )

    if not model_provider:
        raise ValueError(
            "model_provider (or provider) must be specified. "
            f"Supported providers: {', '.join(EMBEDDING_PROVIDERS.keys())}"
        )

    # Normalize provider name
    normalized_provider = model_provider.replace("-", "_").lower()

    # Create base embedding model
    base_embeddings = _create_embedding_model(
        model=model,
        provider=normalized_provider,
        **kwargs,
    )

    # Add caching if requested
    if cache:
        return _create_cached_embeddings(
            base_embeddings=base_embeddings,
            model=model,
            cache_dir=cache_dir,
            cache_namespace=cache_namespace,
            cache_in_memory=cache_in_memory,
        )

    return base_embeddings


def _create_embedding_model(
    model: str | None,
    provider: str,
    **kwargs: Any,
) -> Embeddings:
    """Unified embedding model creation factory function.

    Args:
        model: Model name/identifier
        provider: Normalized provider name
        **kwargs: Additional model parameters

    Returns:
        Embeddings: Initialized embedding model instance

    Raises:
        ValueError: When provider is not supported
        ImportError: When required package is not installed
        AttributeError: When model class is not found
    """
    # Get provider configuration
    config = EMBEDDING_PROVIDERS.get(provider)
    if not config:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {', '.join(EMBEDDING_PROVIDERS.keys())}"
        )

    # Dynamic module import - "Batteries included, but choose the best batteries"
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

    # Build parameters - explicit is better than implicit
    params = _build_embedding_params(config, model, **kwargs)

    return model_class(**params)


def _build_embedding_params(
    config: EmbeddingProviderConfig,
    model: str | None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build embedding model parameters dictionary.

    Args:
        config: Provider configuration
        model: Model name/identifier
        **kwargs: Additional parameters from user

    Returns:
        Dict[str, Any]: Complete parameters dictionary for model initialization
    """
    params = {**kwargs}

    # Add model parameter if provided and not already in kwargs
    if model and "model" not in params:
        params["model"] = model

    # Add API key if configured
    if config.api_key_env:
        api_key = _get_api_key(config.api_key_env)
        if api_key:
            # Handle special parameter names for different providers
            if config.api_key_env == "GOOGLE_API_KEY":
                key_param = "google_api_key"
            elif config.api_key_env == "DASHSCOPE_API_KEY":
                key_param = "dashscope_api_key"
                # DashScope requires plain string, not SecretStr
                api_key = (
                    api_key.get_secret_value()
                    if hasattr(api_key, "get_secret_value")
                    else str(api_key)
                )
            else:
                key_param = "api_key"
            params[key_param] = api_key

    # Add base URL if configured
    if config.base_url_env:
        base_url = _get_env_var(config.base_url_env)
        if base_url:  # Use environment variable if set
            params["base_url"] = base_url
        elif config.default_base_url:  # Use default if env var not set
            params["base_url"] = config.default_base_url

    # Handle special parameters for provider-specific configurations
    if config.special_params:
        for key, value in config.special_params.items():
            if key.endswith("_env"):
                # Environment variable parameters
                env_key = key[:-4]  # Remove _env suffix
                env_value = _get_env_var(value)
                if env_value:
                    params[env_key] = env_value
            else:
                # Direct parameters
                params[key] = value

    # Special handling for Xinference: model parameter should map to model_uid
    if config.class_name == "XinferenceEmbeddings":
        if "model" in params:
            params["model_uid"] = params.pop("model")
        # Ensure server_url is set from environment
        if "server_url" not in params:
            if server_url := _get_env_var("XINFERENCE_SERVER_URL"):
                params["server_url"] = server_url

    return params


def _create_cached_embeddings(
    base_embeddings: Embeddings,
    model: str | None,
    cache_dir: str | Path | None,
    cache_namespace: str | None,
    cache_in_memory: bool,
) -> Embeddings:
    """Create cached embeddings using CacheBackedEmbeddings.

    Args:
        base_embeddings: The underlying embedding model to cache
        model: Model name for default namespace
        cache_dir: Directory for file-based caching
        cache_namespace: Namespace for cache keys
        cache_in_memory: Whether to use in-memory caching

    Returns:
        Embeddings: CacheBackedEmbeddings instance

    Raises:
        ImportError: When required caching packages are not installed
    """
    try:
        from langchain.embeddings import CacheBackedEmbeddings
        from langchain.storage import InMemoryByteStore
        
        # Choose storage type
        store_type = "in-memory"
        
        if cache_in_memory or cache_dir is None:
            # Use in-memory storage
            store = InMemoryByteStore()
            logger.info("Using in-memory embedding cache")
        else:
            # Try to use file-based storage
            try:
                from langchain.storage import LocalFileStore
                # Ensure cache directory exists
                cache_path = Path(cache_dir)
                cache_path.mkdir(parents=True, exist_ok=True)
                store = LocalFileStore(str(cache_path))
                store_type = "file-based"
                logger.info(f"Using file-based embedding cache at: {cache_path}")
            except ImportError:
                # Fall back to in-memory if LocalFileStore is not available
                logger.warning(
                    "LocalFileStore not available in current LangChain version. "
                    "Falling back to in-memory caching. "
                    "Consider upgrading LangChain for persistent file caching."
                )
                store = InMemoryByteStore()
        
        # Determine namespace for cache keys
        if cache_namespace is None:
            # Try to get model name from the embedding instance
            if hasattr(base_embeddings, 'model'):
                cache_namespace = base_embeddings.model
            elif model:
                cache_namespace = model
            else:
                cache_namespace = base_embeddings.__class__.__name__
        
        # Create cached embeddings
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=base_embeddings,
            document_embedding_cache=store,
            namespace=cache_namespace,
            key_encoder="sha256"
        )
        
        logger.info(
            f"Created cached embeddings with namespace '{cache_namespace}' "
            f"using {store_type} storage"
        )
        
        return cached_embeddings
        
    except ImportError as e:
        logger.warning(
            f"Failed to import caching dependencies: {e}. "
            "Falling back to non-cached embeddings. "
            "Install required packages with: pip install langchain"
        )
        return base_embeddings
