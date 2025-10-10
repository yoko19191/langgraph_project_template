"""Utility functions for loading configurations and environment variables."""

import os
from pathlib import Path

try:
    from pydantic import SecretStr
except ImportError:
    SecretStr = str  # type: ignore


def _get_api_key(env_var: str) -> SecretStr | None:
    """Get API key from environment variable.

    Args:
        env_var: Environment variable name

    Returns:
        SecretStr instance or None if not found
    """
    api_key = os.getenv(env_var)
    return SecretStr(api_key) if api_key else None


def _get_env_var(env_var: str) -> str | None:
    """Get environment variable value.

    Args:
        env_var: Environment variable name

    Returns:
        Variable value or None if not found
    """
    value = os.getenv(env_var)
    return value if value else None


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path object pointing to the project root.
    """
    # Assuming this file is at src/app/common/utils/load.py
    # Go up 4 levels to reach project root
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    return project_root
