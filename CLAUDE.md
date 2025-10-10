# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **production-ready FastAPI template for scalable agentic applications** built on LangGraph. The project emphasizes extensibility and Pythonic design principles, with comprehensive support for multiple AI model providers, MCP (Model Context Protocol) integration, and enterprise-grade logging capabilities.

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
pytest tests/app/common/tools/test_mcp.py # Run MCP tests
pytest tests/app/common/utils/test_logging.py # Run logging tests
pytest -k "test_openai"        # Run tests matching pattern
pytest -v                     # Verbose output
PYTHONPATH=. python tests/test_mcp_config.py # Direct test execution
```

### LangGraph Development
```bash
# LangGraph CLI (if graphs are added)
langgraph dev                  # Start development server
langgraph build                # Build graphs
```

## Core Architecture

### Chat Model System (`src/app/common/models/chat_models.py`)

The project features a **sophisticated config-driven chat model system** supporting 20+ providers:

**Key Architecture Principles:**
- **Config-driven design**: All providers defined in `PROVIDERS` dictionary using `ProviderConfig` dataclass
- **Factory pattern**: Unified `init_chat_model()` function handles all providers
- **Intelligent routing**: Automatic provider inference and alias support
- **Error resilience**: Graceful fallback to langchain defaults

**Supported Providers:**
- **International**: OpenAI, Anthropic, Google, Groq, DeepSeek
- **Chinese**: Dashscope (Qwen), ZhipuAI (GLM), Moonshot (Kimi), Volcengine (Doubao), SiliconFlow
- **Custom**: OpenRouter, DMXAPI, Xinference, Ollama

**Usage Examples:**
```python
from src.app.common.models.chat_models import init_chat_model

# Auto-inference
model = init_chat_model("qwen-max")  # -> dashscope provider

# Explicit provider
model = init_chat_model("gpt-4", model_provider="openai")

# Alias format
model = init_chat_model("opr:anthropic/claude-3-haiku")  # OpenRouter alias
```

### Enhanced MCP Integration (`src/app/common/tools/mcp.py`)

**Comprehensive Multi-Server MCP Client with Smart Configuration:**

**Key Features:**
- **Smart config discovery**: Automatically finds `mcp.json` by traversing up directory tree
- **Full MCP protocol support**: Tools, prompts, and resources
- **Async/sync APIs**: Both `aget_*` and `get_*` methods available
- **Per-server caching**: Intelligent caching with server isolation
- **Runtime management**: Add/remove servers, clear caches

**Core APIs:**
```python
from src.app.common.tools.mcp import (
    get_mcp_tools, aget_mcp_tools,           # Server-specific tools
    get_all_mcp_tools, aget_all_mcp_tools,   # All tools from all servers
    get_mcp_prompts, aget_mcp_prompts,       # Server prompts
    get_mcp_resources, aget_mcp_resources,   # Server resources
    add_mcp_server, remove_mcp_server,       # Runtime management
    clear_mcp_cache                          # Cache management
)

# Usage examples
tools = get_mcp_tools("context7")           # Get tools from specific server
all_tools = get_all_mcp_tools()             # Get all tools from all servers
prompts = get_mcp_prompts("deepwiki")       # Get prompts from specific server
```

**MCP Configuration (`mcp.json`):**
```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@context7/server@latest"]
    },
    "deepwiki": {
      "command": "npx", 
      "args": ["-y", "@deepwiki/mcp-server@latest"]
    }
  }
}
```

### Centralized Logging System (`src/app/common/utils/logging.py`)

**Enterprise-grade logging with environment variable configuration:**

**Key Features:**
- **Environment-driven**: `LOG_LEVEL` env var support (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **File logging**: Optional timestamp-based log files in `logs/` directory
- **Runtime control**: Change log levels at runtime
- **Third-party silencing**: Disable noisy library loggers
- **Flexible output**: Console, file, or both

**Core APIs:**
```python
from src.app.common.utils.logging import (
    setup_logging,     # Main setup function
    get_logger,        # Get named logger
    set_log_level,     # Runtime level changes
    disable_logger     # Silence noisy loggers
)

# Setup examples
setup_logging()                                    # Basic console logging
setup_logging(save_log=True)                      # Console + file logging
setup_logging(save_log=True, console_output=False) # File-only logging

# Usage
logger = get_logger(__name__)
logger.info("Application started")

# Runtime control
set_log_level("DEBUG")              # Change level at runtime
disable_logger("urllib3")           # Silence noisy libraries
```

**Environment Configuration:**
```bash
# .env file
LOG_LEVEL=DEBUG  # Will be used by setup_logging()
```

### Smart Configuration Management (`src/app/common/utils/mcp.py`)

**Intelligent config file discovery and validation:**

**Key Features:**
- **Smart discovery**: `find_nearest_config_file()` traverses directory tree upward
- **Configurable filenames**: Not hardcoded to "mcp.json"
- **Pydantic validation**: Type-safe config loading with `MCPConfig` model
- **Safe JSON parsing**: Secure config file loading

**Core APIs:**
```python
from src.app.common.utils.mcp import find_nearest_config_file, load_mcp_config

# Find config file anywhere in project hierarchy
config_path = find_nearest_config_file("mcp.json")

# Load with custom name and validation
config = load_mcp_config("custom-mcp.json")
```

### Project Structure
```
src/app/
├── __init__.py
├── app.py                     # Main FastAPI application (ready for implementation)
└── common/
    ├── models/
    │   ├── chat_models.py     # Core chat model provider system
    │   ├── embedding_models.py # Embedding model support
    │   └── document_compressor.py # Document reranking
    ├── tools/
    │   └── mcp.py             # Enhanced MCP client with full protocol support
    ├── utils/
    │   ├── logging.py         # Centralized logging system
    │   ├── mcp.py             # Smart MCP configuration management
    │   └── load.py            # Core utilities
    └── prebuild/              # Pre-built components (agents, workflows)
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

### MCP Server Integration

**Adding new MCP servers:**
1. **Add to `mcp.json`** (can be placed anywhere in project hierarchy):
```json
{
  "mcpServers": {
    "new-server": {
      "command": "path/to/executable",
      "args": ["arg1", "arg2"]
    }
  }
}
```

2. **Use in code**:
```python
from src.app.common.tools.mcp import get_mcp_tools, get_mcp_prompts

tools = get_mcp_tools("new-server")      # Get tools from new server
prompts = get_mcp_prompts("new-server")  # Get prompts from new server
```

### Application Logging Setup

**For new modules, follow this pattern:**
```python
from src.app.common.utils.logging import get_logger

logger = get_logger(__name__)  # Use module name

def your_function():
    logger.info("Function started")
    try:
        # Your logic here
        logger.debug("Debug information")
        return result
    except Exception as e:
        logger.error(f"Function failed: {e}", exc_info=True)
        raise
```

**For application startup:**
```python
from src.app.common.utils.logging import setup_logging, disable_logger

# Setup logging early in application lifecycle
setup_logging(
    save_log=True,                          # Enable file logging
    log_level=os.getenv("LOG_LEVEL", "INFO"), # Use env var
    console_output=True                     # Keep console output
)

# Silence noisy third-party libraries
for lib in ["urllib3", "httpx", "httpcore"]:
    disable_logger(lib)
```

### Testing Strategy

**Comprehensive test coverage:**
- **MCP Tests** (`tests/app/common/tools/test_mcp.py`): 21+ tests covering sync/async operations, caching, error handling
- **Logging Tests** (`tests/app/common/utils/test_logging.py`): 19+ tests covering setup, file output, env vars, runtime changes
- **Chat Model Tests** (`tests/test_chat_model.py`): Provider inference, configuration, API integration

**Running specific test suites:**
```bash
# MCP functionality
PYTHONPATH=. pytest tests/app/common/tools/test_mcp.py -v

# Logging functionality  
PYTHONPATH=. pytest tests/app/common/utils/test_logging.py -v

# Core chat models
pytest tests/test_chat_model.py::TestModelParsing -v
```

### Environment Configuration

**Multi-Provider Setup with Enhanced Logging:**
- **Package management**: Uses Chinese PyPI mirrors for faster installation
- **API Provider Support**: 20+ environment variables for different providers
- **Observability**: LangSmith tracing + comprehensive application logging
- **MCP Integration**: Smart config discovery with validation
- **Development Tools**: uv, ruff, pytest with async support

## Key Technical Details

### Provider Configuration System
- **Type-safe**: Uses `@dataclass` with proper type hints
- **Flexible parameter building**: Handles API keys, base URLs, and special parameters  
- **Dynamic class selection**: Special logic for models like QwQ vs Qwen
- **Graceful error handling**: Clear error messages for missing packages/classes

### MCP Architecture
- **Server isolation**: Each server gets its own client instance for tool loading
- **Smart caching**: Per-server caches with automatic invalidation
- **Full protocol support**: Tools, prompts, resources with async/sync APIs
- **Runtime management**: Add/remove servers without restart

### Logging Architecture
- **Environment-driven**: LOG_LEVEL env var automatically detected
- **Flexible output**: Console, file, or both with independent configuration
- **Timestamp-based naming**: Log files use `app_YYYYMMDD_HHMMSS.log` format
- **Project-aware paths**: Automatically detects project root for log directory

### Code Quality Standards
- **Ruff linting**: Configured for pycodestyle, pyflakes, isort, pydocstyle
- **Google docstring convention**: Enforced via ruff configuration
- **Python 3.12+**: Modern Python features and type hints
- **Comprehensive testing**: 40+ tests with async support and automatic API key handling

This architecture enables rapid development of production-ready agentic applications while maintaining enterprise-grade code quality, observability, and extensibility.