from __future__ import annotations

import warnings

from typing import (
    Any,    
    Union,
    Literal,
    Optional,
    cast
)

from app.common.utils import (
    _get_api_key, 
    _get_env_var, 
    _check_pkg
)

from langchain_core.language_models import (
    BaseChatModel,
)
from langchain_community.chat_models import VolcEngineMaasChat

from langchain.chat_models.base import (
    _ConfigurableModel
)

from langchain.chat_models import (
    init_chat_model as langchain_chat_model
)
from langchain.chat_models.base import (
    _attempt_infer_model_provider as langchain_infer_provider
)


def init_chat_model(
    model: Optional[str]=None,
    *,
    model_provider: Optional[str]=None,
    configurable_fields: Optional[
        Union[Literal["any"], list[str], tuple[str, ...]]
    ]=None,
    config_prefix: Optional[str] = None,
    **kwargs: Any
) -> Union[BaseChatModel, _ConfigurableModel]:
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
        return _init_chat_model_helper(
            cast(str, model),
            model_provider=model_provider,
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
    
def _init_chat_model_helper(
    model: Optional[str] = None,
    *,
    model_provider: Optional[str] = None,
    **kwargs: Any,
) -> BaseChatModel:
    model, model_provider = _parse_model(model, model_provider)
    if model_provider in ("openrouter", "opr"):
        _check_pkg("langchain-openai")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=_get_api_key("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            **kwargs
        )
    elif model_provider in ("dashscope", "qwen"):
        _check_pkg("langchain_qwq")
        from langchain_qwq import ChatQwen, ChatQwQ
        # 获取 API key，如果没有则使用默认值
        api_key = _get_api_key("DASHSCOPE_API_KEY")
        base_url = _get_env_var("DASHSCOPE_API_BASE")
        if model.startswith(("qwq", "qvq")):
            return ChatQwQ(model=model, api_key=api_key, base_url=base_url, **kwargs)
        else:
            return ChatQwen(model=model, api_key=api_key, base_url=base_url, **kwargs)
    elif model_provider in ("zhipuai"):
        _check_pkg("langchain_community")
        from langchain_community.chat_models import ChatZhipuAI
        return ChatZhipuAI(
            model=model,
            **kwargs
        )
    elif model_provider in ("moonshot"):
        _check_pkg("langchain-openai")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=_get_api_key("MOONSHOT_API_KEY"),
            base_url=_get_env_var("MOONSHOT_BASE_URL"),
            **kwargs
        )
    elif model_provider in ("volcengine", "ark"):
        _check_pkg("langchain-openai")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=_get_api_key("ARK_API_KEY"),
            base_url=_get_env_var("ARK_BASE_URL"),
            **kwargs
        )
    elif model_provider == "siliconflow":
        _check_pkg("langchain_siliconflow")
        from langchain_siliconflow import ChatSiliconFlow
        return ChatSiliconFlow(
            model=model,
            api_key=_get_api_key("SILICONFLOW_API_KEY"),
            base_url=_get_env_var("SILICONFLOW_BASE_URL"),
            **kwargs
        )
    elif model_provider == "dmxapi":
        _check_pkg("langchain-openai")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=_get_api_key("DMXAPI_API_KEY"),
            base_url=_get_env_var("DMXAPI_BASE_URL"),
            **kwargs
        )
    elif model_provider == "xinference":
        _check_pkg("langchain-xinference")
        from langchain_xinference import ChatXinference
        return ChatXinference(
            server_url=_get_env_var("XINFERENCE_SERVER_URL"),
            model_uid=_get_env_var("XINFERENCE_CHAT_MODEL_UID"),
            **kwargs
        )
    elif model_provider == "google":
        _check_pkg("langchain-google-genai")
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # 构建 client_options
        client_options = {}
        if api_endpoint := _get_env_var("GOOGLE_API_ENDPOINT"):
            client_options["api_endpoint"] = api_endpoint
            
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=_get_api_key("GOOGLE_API_KEY"),
            client_options=client_options if client_options else None,
            **kwargs
        )
        
    else:
        return langchain_chat_model(model=model, model_provider=model_provider, **kwargs)

        
_SUPPORTED_PROVIDERS = [
    # parsing langchain init_chat_model
    "openai",
    "anthropic",
    "ollama",
    "groq",
    "deepseek",
    # custome parsing
    "xinference",
    "google_genai",
    # openai compatible
    "openrouter",
    "dashscope",
    "zhipuai",
    "moonshot",
    "volcengine",
    "siliconflow",
    "dmxapi.cn"
]


def _parse_model(model: Optional[str], model_provider: Optional[str]) -> tuple[str, str]:
    if not model:
        raise ValueError("model parameter is required")
        
    if (
        not model_provider
        and ":" in model
        and model.split(":")[0] in _SUPPORTED_PROVIDERS
    ):
        model_provider = model.split(":")[0]
        model = ":".join(model.split(":")[1:])
    model_provider = model_provider or _attempt_infer_model_provider(model)
    if not model_provider:
        msg = (
            f"Unable to infer model provider for {model=}, please specify "
            f"model_provider directly."
        )
        raise ValueError(msg)
    
    
    model_provider = model_provider.replace("-", "_").lower()
    return model, model_provider


def _attempt_infer_model_provider(model_name: str) -> Optional[str]:
    """attempt infer additional model provider
    Args:
        model_name: model name
        
    Returns:
        inferred model provider name, if not inferred, return None
    """
    # 
    if any(model_name.startswith(pre) for pre in ("qwen", "qwq", "qvq")):
        return "dashscope"
    if any(model_name.startswith(pre) for pre in ("kimi", "moonshot")):
        return "moonshot"
    if model_name.startswith("doubao"):
        return "volcengine"
    if model_name.startswith("glm"):
        return "zhipuai"
    if any(model_name.startswith(pre) for pre in ("gemini", "gemma")):
        return "google"
    
    # Anything else fall back to langchain's default provider inference
    return langchain_infer_provider(model_name)
