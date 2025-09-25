# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **production-ready FastAPI template for scalable agentic applications** built on LangGraph. The project emphasizes extensibility and Pythonic design principles, with comprehensive support for multiple AI model providers and MCP (Model Context Protocol) integration.

## Essential Commands

### Package Management (uv)
```bash
# Initial setup
uv sync                          # Install all dependencies
cp .env.example .env            # Setup environment variables
uv add package-name             # Add new dependency
uv add --dev package-name       # Add development dependency
```

### Development & Quality
```bash
# Code quality
ruff check                      # Run linter
ruff check --fix               # Auto-fix linting issues
ruff format                    # Format code

# Testing
pytest                         # Run all tests
pytest tests/test_chat_model.py # Run specific test file
pytest -k "test_openai"        # Run tests matching pattern
pytest -v                     # Verbose output
python tests/test_chat_model.py # Run basic functionality tests
```

### LangGraph Development
```bash
# LangGraph CLI (if graphs are added)
langgraph dev                  # Start development server
langgraph build                # Build graphs
```

## Core Architecture

### Chat Model System (`src/app/common/models/chat_models.py`)

The project features a **sophisticated config-driven chat model system** supporting 9+ providers:

**Key Architecture Principles:**
- **Config-driven design**: All providers defined in `PROVIDERS` dictionary using `ProviderConfig` dataclass
- **Factory pattern**: Unified `_create_chat_model()` function handles all providers
- **Intelligent routing**: Automatic provider inference and alias support
- **Error resilience**: Graceful fallback to langchain defaults

**Supported Providers:**
- International: OpenAI, Anthropic, Google, Groq, DeepSeek
- Chinese: Dashscope (Qwen), ZhipuAI (GLM), Moonshot (Kimi), Volcengine (Doubao), SiliconFlow
- Custom: OpenRouter, DMXAPI, Xinference

**Usage Examples:**
```python
from app.common.models.chat_models import init_chat_model

# Auto-inference
model = init_chat_model("qwen-max")  # -> dashscope provider

# Explicit provider
model = init_chat_model("gpt-4", model_provider="openai")

# Alias format
model = init_chat_model("opr:anthropic/claude-3-haiku")  # OpenRouter alias
```

### MCP Integration (`src/app/common/mcp_client.py`)

**Multi-Server MCP Setup:**
- Global client management with caching
- Server configurations in `mcp.json`
- Currently integrated: jina-mcp, deepwiki, context7

**Configuration Pattern:**
```json
{
  "mcpServers": {
    "server-name": {
      "url": "server-url",
      "command": "executable",
      "args": ["arg1", "arg2"]
    }
  }
}
```

### Project Structure
```
src/app/
├── __init__.py
├── app.py                     # Main FastAPI application (empty - ready for implementation)
└── common/
    ├── models/
    │   ├── chat_models.py     # Core chat model provider system
    │   └── embedding_models.py # Embedding model imports
    ├── utils.py               # Shared utilities
    └── mcp_client.py          # MCP client management
```

## Development Workflows

### Adding New Chat Model Providers

1. **Add provider configuration** in `PROVIDERS` dict:
```python
"new_provider": ProviderConfig(
    package="langchain-package-name",
    module="langchain_module_name",
    class_name="ChatModelClass",
    api_key_env="API_KEY_ENV_VAR",
    base_url_env="BASE_URL_ENV_VAR"  # optional
)
```

2. **Add environment variables** to `.env.example`:
```bash
NEW_PROVIDER_API_KEY="your-api-key"
NEW_PROVIDER_BASE_URL="https://api.provider.com/v1"  # if needed
```

3. **Add provider inference** in `_attempt_infer_model_provider()` if needed:
```python
if model_name.startswith("provider-prefix"):
    return "new_provider"
```

4. **Add to supported providers list** for parsing:
```python
_SUPPORTED_PROVIDERS = [..., "new_provider", "alias"]
```

### Testing Strategy

**Comprehensive test coverage in `tests/test_chat_model.py`:**
- Model parsing and provider inference
- Provider normalization and aliases
- Configuration system validation
- API-key dependent tests (automatically skipped if keys missing)

**Running provider-specific tests:**
```bash
# Test specific provider (will skip if no API key)
pytest tests/test_chat_model.py::TestCustomProviders::test_openrouter_model

# Test core functionality (no API keys needed)
pytest tests/test_chat_model.py::TestModelParsing
```

### Environment Configuration

**Multi-Provider Setup with Chinese Mirror Support:**
- **uv configuration**: Uses Chinese PyPI mirrors for faster package installation
- **API Provider Support**: 20+ environment variables for different providers
- **Observability**: LangSmith tracing integration configured
- **Database/Storage**: Prepared slots for Postgres, Redis, Milvus, OpenDAL

### MCP Server Integration

**Adding new MCP servers:**
1. Add to `mcp.json` configuration
2. Import in `mcp_client.py` if custom handling needed
3. Use `MultiServerMCPClient` for tool access

## Key Technical Details

### Provider Configuration System
- **Type-safe**: Uses `@dataclass` with proper type hints
- **Flexible parameter building**: Handles API keys, base URLs, and special parameters
- **Dynamic class selection**: Special logic for models like QwQ vs Qwen
- **Graceful error handling**: Clear error messages for missing packages/classes

### Code Quality Standards
- **Ruff linting**: Configured for pycodestyle, pyflakes, isort, pydocstyle
- **Google docstring convention**: Enforced via ruff configuration
- **Python 3.12+**: Modern Python features and type hints
- **Test coverage**: Comprehensive pytest suite with async support

This architecture enables rapid development of agentic applications while maintaining code quality and extensibility.