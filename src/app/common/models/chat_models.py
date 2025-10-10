"""Chat model initialization module.

Enhanced version of langchain init_chat_model with support for more Chinese local model providers.
"""

from __future__ import annotations

import importlib
import logging
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import (
    Any,
    Dict,
    Literal,
    Union,
)

from langchain.chat_models import init_chat_model as langchain_chat_model
from langchain.chat_models.base import (
    _attempt_infer_model_provider as langchain_infer_provider,
)
from langchain.chat_models.base import _ConfigurableModel
from langchain_core.language_models import (
    BaseChatModel,
)

from src.app.common.utils.load import _get_api_key, _get_env_var

logger = logging.getLogger(__name__)

# Type aliases for better readability - "Readability counts"
type ChatModelType = BaseChatModel
type ParamsDict = Dict[str, Any]


@dataclass
class ProviderConfig:
    """Chat model provider configuration.

    This dataclass encapsulates all provider-specific settings,
    making it easy to add new providers without code changes.

    Attributes:
        package: Required package name (e.g., "langchain-openai")
        module: Module to import (e.g., "langchain_openai")
        class_name: Default model class name
        api_key_env: Environment variable for API key
        base_url_env: Environment variable for base URL
        default_base_url: Fallback base URL when env var is not set
        model_prefixes: Tuple of model name prefixes for auto-inference
        special_params: Additional provider-specific parameters
    """

    package: str
    module: str
    class_name: str
    api_key_env: str | None = None
    base_url_env: str | None = None
    default_base_url: str | None = None
    model_prefixes: tuple[str, ...] | None = None  # New: for config-driven inference
    special_params: Dict[str, Any] | None = None


# Configuration-driven provider registry - "There should be one obvious way to do it"
PROVIDERS: Dict[str, ProviderConfig] = {
    "openrouter": ProviderConfig(
        package="langchain-openai",
        module="langchain_openai",
        class_name="ChatOpenAI",
        api_key_env="OPENROUTER_API_KEY",
        model_prefixes=("opr:",),  # OpenRouter uses prefix format
        special_params={"base_url": "https://openrouter.ai/api/v1"},
    ),
    "dashscope": ProviderConfig(
        package="langchain-community",
        module="langchain_qwq",
        class_name="ChatQwen",  # Default class, can be dynamically replaced
        api_key_env="DASHSCOPE_API_KEY",
        base_url_env="DASHSCOPE_API_BASE",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_prefixes=("qwen-", "qwq-", "qvq-"),
    ),
    "zhipuai": ProviderConfig(
        package="langchain-community",
        module="langchain_community.chat_models",
        class_name="ChatZhipuAI",
        model_prefixes=("glm-",),
    ),
    "moonshot": ProviderConfig(
        package="langchain-openai",
        module="langchain_openai",
        class_name="ChatOpenAI",
        api_key_env="MOONSHOT_API_KEY",
        base_url_env="MOONSHOT_BASE_URL",
        default_base_url="https://api.moonshot.cn/v1",
        model_prefixes=("moonshot-", "kimi-"),
    ),
    "volcengine": ProviderConfig(
        package="langchain-openai",
        module="langchain_openai",
        class_name="ChatOpenAI",
        api_key_env="ARK_API_KEY",
        base_url_env="ARK_BASE_URL",
        default_base_url="https://ark.cn-beijing.volces.com/api/v3",
        model_prefixes=("doubao-",),
    ),
    "siliconflow": ProviderConfig(
        package="langchain-siliconflow",
        module="langchain_siliconflow",
        class_name="ChatSiliconFlow",
        api_key_env="SILICONFLOW_API_KEY",
        base_url_env="SILICONFLOW_BASE_URL",
        default_base_url="https://api.siliconflow.cn/v1",
        model_prefixes=("sf-",),
    ),
    "dmxapi": ProviderConfig(
        package="langchain-openai",
        module="langchain_openai",
        class_name="ChatOpenAI",
        api_key_env="DMXAPI_API_KEY",
        base_url_env="DMXAPI_BASE_URL",
        default_base_url="https://www.DMXapi.cn/v1",
        model_prefixes=(),  # No specific prefix, explicit provider needed
    ),
    "xinference": ProviderConfig(
        package="langchain-xinference",
        module="langchain_xinference",
        class_name="ChatXinference",
        model_prefixes=("xinference-",),
        special_params={
            "server_url_env": "XINFERENCE_SERVER_URL",
            "model_uid_env": "XINFERENCE_CHAT_MODEL_UID",
        },
    ),
    "google": ProviderConfig(
        package="langchain-google-genai",
        module="langchain_google_genai",
        class_name="ChatGoogleGenerativeAI",
        api_key_env="GOOGLE_API_KEY",
        model_prefixes=("gemini-", "gemma-"),
    ),
}


def init_chat_model(
    model: str | None = None,
    *,
    model_provider: str | None = None,
    configurable_fields: (Literal["any"] | list[str] | tuple[str, ...] | None) = None,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> Union[BaseChatModel, _ConfigurableModel]:
    """Initialize chat model with enhanced provider support.

    This is a wrapper around langchain's native init_chat_model with support for additional providers.

    Supported providers:
    - Standard langchain providers: openai, anthropic, ollama, groq, deepseek
    - Chinese providers: dashscope/qwen, zhipuai, moonshot, volcengine/ark, siliconflow, dmxapi
    - Custom providers: xinference, google, openrouter

    Args:
        model: Model name, supports "provider:model" format
        model_provider: Explicitly specified model provider
        configurable_fields: Configurable fields
        config_prefix: Configuration prefix
        **kwargs: Additional model parameters

    Returns:
        Chat model instance or configurable model

    Raises:
        ValueError: When model name is empty or provider cannot be inferred
    """
    if not model and not configurable_fields:
        configurable_fields = ("model", "model_provider")
    config_prefix = config_prefix or ""
    if config_prefix and not configurable_fields:
        warnings.warn(
            f"{config_prefix=} has been set but no fields are configurable. Set "
            f"`configurable_fields=(...)` to specify the model params that are "
            f"configurable.",
            stacklevel=2,
        )
    if not configurable_fields:
        parsed_model, parsed_provider = _parse_model(model, model_provider)
        return _create_chat_model(
            parsed_model,
            parsed_provider,
            **kwargs,
        )
    if model:
        kwargs["model"] = model
    if model_provider:
        kwargs["model_provider"] = model_provider
    return _ConfigurableModel(
        default_config=kwargs,
        config_prefix=config_prefix,
        configurable_fields=configurable_fields,
    )


@lru_cache(maxsize=32)
def _import_model_class(module_path: str, class_name: str) -> type[ChatModelType]:
    """Import and cache model class to avoid repeated imports.

    Performance optimization: cache frequently used model classes.

    Args:
        module_path: Full module path to import from
        class_name: Name of the class to import

    Returns:
        The imported model class

    Raises:
        ImportError: If module cannot be imported
        AttributeError: If class not found in module
    """
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module {module_path}. "
            f"Please install the required package: pip install <package>"
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Class {class_name} not found in module {module_path}"
        ) from e


def _create_chat_model(
    model: str,
    model_provider: str,
    **kwargs: Any,
) -> ChatModelType:
    """Unified chat model creation factory function.

    Args:
        model: Model name
        model_provider: Provider name
        **kwargs: Additional model parameters

    Returns:
        Initialized chat model instance
    """
    # Normalize provider name
    normalized_provider = _normalize_provider_name(model_provider)

    # Check if there's a custom configuration
    config = PROVIDERS.get(normalized_provider)
    if not config:
        # Fallback to langchain default implementation
        return langchain_chat_model(
            model=model, model_provider=model_provider, **kwargs
        )

    # Handle special class selection logic (e.g., dashscope's QwQ vs Qwen)
    class_name = _get_model_class_name(config, model)

    # Import model class with caching
    model_class = _import_model_class(config.module, class_name)

    # Build parameters - explicit is better than implicit
    params = _build_model_params(config, model, **kwargs)

    return model_class(**params)


def _normalize_provider_name(provider: str) -> str:
    """Normalize provider name."""
    # Handle alias mapping
    alias_map = {"opr": "openrouter", "qwen": "dashscope", "ark": "volcengine"}
    return alias_map.get(provider, provider.replace("-", "_").lower())


def _get_model_class_name(config: ProviderConfig, model: str) -> str:
    """Determine the class to use based on model name."""
    # Special handling for dashscope's QwQ models
    if config.class_name == "ChatQwen" and model and model.startswith(("qwq", "qvq")):
        return "ChatQwQ"
    return config.class_name


def _build_model_params(
    config: ProviderConfig, model: str, **kwargs: Any
) -> ParamsDict:
    """Build model initialization parameters from config and kwargs.

    Args:
        config: Provider configuration
        model: Model name
        **kwargs: Additional parameters to merge

    Returns:
        Dictionary of parameters for model initialization
    """
    params: ParamsDict = {"model": model, **kwargs}

    # Add API key
    if config.api_key_env:
        api_key = _get_api_key(config.api_key_env)
        # Handle special parameter names (e.g., Google's google_api_key)
        key_param = (
            "google_api_key" if config.api_key_env == "GOOGLE_API_KEY" else "api_key"
        )
        params[key_param] = api_key

    # Add base URL
    if config.base_url_env:
        base_url = _get_env_var(config.base_url_env)
        if base_url:  # Use environment variable if set
            params["base_url"] = base_url
        elif config.default_base_url:  # Use default if env var not set
            params["base_url"] = config.default_base_url

    # Handle special parameters
    if config.special_params:
        for key, value in config.special_params.items():
            if key.endswith("_env"):
                # Environment variable parameters
                env_key = key[:-4]  # Remove _env suffix
                if env_value := _get_env_var(value):
                    if env_key == "server_url":
                        params["server_url"] = env_value
                    elif env_key == "model_uid":
                        params["model_uid"] = env_value
            else:
                # Direct parameters
                params[key] = value

    # Special handling for Xinference: model parameter should map to model_uid
    if config.class_name == "ChatXinference":
        params["model_uid"] = params.pop("model")  # Change model to model_uid
        # Get server_url from environment variable
        if server_url := _get_env_var("XINFERENCE_SERVER_URL"):
            params["server_url"] = server_url

    return params


_SUPPORTED_PROVIDERS = [
    # parsing langchain init_chat_model
    "openai",
    "anthropic",
    "ollama",
    "groq",
    "deepseek",
    # custome parsing
    "xinference",
    "google",
    # openai compatible (including aliases)
    "openrouter",
    "opr",
    "dashscope",
    "qwen",
    "zhipuai",
    "moonshot",
    "volcengine",
    "ark",
    "siliconflow",
    "dmxapi",
]


def _parse_model(model: str | None, model_provider: str | None) -> tuple[str, str]:
    """Parse model name and provider.

    Embodies Python Zen: Explicit is better than implicit, errors should never pass silently.
    """
    if not model:
        raise ValueError("Model name cannot be empty")

    # Handle "provider:model" format - use safer splitting method
    if model_provider is None and ":" in model:
        provider_part, model_part = model.split(":", 1)  # Split only once, safer
        if provider_part in _SUPPORTED_PROVIDERS:
            return model_part, provider_part

    # Try to infer provider
    inferred_provider = model_provider or _attempt_infer_model_provider(model)
    if not inferred_provider:
        raise ValueError(
            f"Cannot infer provider for model '{model}', please explicitly specify model_provider parameter"
        )

    # Normalize provider name
    normalized_provider = inferred_provider.replace("-", "_").lower()
    return model, normalized_provider


def _attempt_infer_model_provider(model_name: str) -> str | None:
    """Infer provider from model name using configured prefixes.

    Configuration-driven inference - "There should be one obvious way to do it".
    This eliminates 20+ if-else statements by using the model_prefixes configuration.

    Args:
        model_name: Model name to infer provider from

    Returns:
        Inferred provider name, or None if no match found

    Examples:
        >>> _attempt_infer_model_provider("qwen-max")
        'dashscope'
        >>> _attempt_infer_model_provider("glm-4")
        'zhipuai'
        >>> _attempt_infer_model_provider("moonshot-v1")
        'moonshot'
    """
    model_lower = model_name.lower()

    # Configuration-driven inference - no if-else chains needed!
    for provider_name, config in PROVIDERS.items():
        if config.model_prefixes:
            if any(model_lower.startswith(prefix) for prefix in config.model_prefixes):
                logger.debug(
                    f"Inferred provider '{provider_name}' from model '{model_name}'"
                )
                return provider_name

    # Fallback to langchain default inference for standard providers
    logger.debug(
        f"No custom provider match for '{model_name}', trying langchain inference"
    )
    return langchain_infer_provider(model_name)
