import os

try:
    from pydantic import SecretStr
except ImportError:
    SecretStr = str  # type: ignore

def _get_api_key(env_var: str) -> SecretStr | None:
    """获取环境变量中的 API key."""
    api_key = os.getenv(env_var)
    return SecretStr(api_key) if api_key else None


def _get_env_var(env_var: str) -> str | None:
    """获取环境变量."""
    value = os.getenv(env_var)
    return value if value else None

