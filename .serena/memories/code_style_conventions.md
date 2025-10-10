# 代码风格和规范

## Python 版本和类型系统

### Python 版本
- **要求**: Python 3.12+
- **现代特性**: 使用 3.12+ 的类型联合语法 `str | None` 而不是 `Optional[str]`

### 类型注解规范
```python
# ✅ 推荐：现代 Python 3.12+ 语法
def process(data: str | None) -> dict[str, Any]:
    pass

# ❌ 避免：旧版语法
from typing import Optional, Dict
def process(data: Optional[str]) -> Dict[str, Any]:
    pass
```

## 代码风格工具链

### Ruff 配置
项目使用 Ruff 作为唯一的代码质量工具（linter + formatter）：

**启用的检查规则**：
- `E`: pycodestyle 错误
- `F`: pyflakes
- `I`: isort（导入排序）
- `D`: pydocstyle（文档字符串）
- `D401`: 首行必须使用祈使句
- `T201`: print 语句检查
- `UP`: pyupgrade（现代化语法）

**忽略的规则**：
- `UP006`, `UP007`: 保留 typing_extensions 导入
- `UP035`: 允许从 typing_extensions 导入
- `D417`: 放宽参数文档要求
- `E501`: 放宽行长度限制

**测试文件特例**：
- `tests/*`: 忽略 `D`（文档字符串）和 `UP`（语法升级）

### 文档字符串规范
使用 **Google 风格文档字符串**：

```python
def init_chat_model(
    model: str | None = None,
    *,
    model_provider: str | None = None,
    **kwargs: Any
) -> BaseChatModel:
    """Initialize chat model with enhanced provider support.

    This is a wrapper around langchain's native init_chat_model with 
    support for additional providers.
    
    Supported providers:
    - Standard: openai, anthropic, ollama, groq, deepseek
    - Chinese: dashscope/qwen, zhipuai, moonshot, siliconflow
    - Custom: xinference, google, openrouter

    Args:
        model: Model name, supports "provider:model" format
        model_provider: Explicitly specified model provider
        **kwargs: Additional model parameters

    Returns:
        Chat model instance or configurable model

    Raises:
        ValueError: When model name is empty or provider cannot be inferred
    """
    pass
```

**关键要求**：
1. 首行必须以祈使句开头（D401）
2. 使用 Args/Returns/Raises 标准格式
3. 简洁描述功能，而非实现细节
4. 公共 API 必须有文档，内部函数可选

## 命名规范

### 模块和包
- 小写字母，下划线分隔：`chat_models.py`, `embedding_models.py`
- 包名简短有意义：`models/`, `tools/`, `utils/`

### 类名
- 大驼峰命名（PascalCase）：`ProviderConfig`, `BaseChatModel`
- 清晰表达用途：`EmbeddingProviderConfig`

### 函数和变量
- 小写字母，下划线分隔：`init_chat_model`, `model_provider`
- 私有函数加前缀：`_create_chat_model`, `_parse_model`
- 布尔变量使用 is/has 前缀（可选但推荐）

### 常量
- 全大写，下划线分隔：`PROVIDERS`, `EMBEDDING_PROVIDERS`
- 私有常量加前缀：`_SUPPORTED_PROVIDERS`

## 代码组织原则

### 导入顺序（isort 自动处理）
```python
# 1. 标准库
import logging
from dataclasses import dataclass
from typing import Any

# 2. 第三方库
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

# 3. 本地导入
from app.common.utils import load_env
```

### 函数组织
- 公共 API 在前
- 私有辅助函数在后
- 相关函数分组，用注释分隔

### 文件长度
- 单个文件不超过 500 行（建议）
- 超过则考虑拆分模块

## 设计模式应用

### 1. 工厂模式（Factory）
用于模型创建：
```python
# ProviderConfig + _create_chat_model
# 配置驱动，易扩展
```

### 2. 配置类（Dataclass）
```python
@dataclass
class ProviderConfig:
    """清晰的配置结构，类型安全"""
    package: str
    module: str
    class_name: str
    api_key_env: str | None = None
```

### 3. 策略模式
- 使用字典映射而非 if-else 链
- `PROVIDERS` 字典即策略集合

## 错误处理规范

### 明确的异常类型
```python
# ✅ 明确具体的异常
if not model:
    raise ValueError("Model name is required")

# ❌ 避免通用异常
if not model:
    raise Exception("Error")
```

### 日志记录
```python
# 使用标准库 logging
import logging
logger = logging.getLogger(__name__)

# 不同级别
logger.debug("详细调试信息")
logger.info("常规信息")  
logger.warning("警告")
logger.error("错误")
```

## 测试规范

### Pytest 配置
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
```

### 测试文件组织
- 镜像 src 结构：`tests/app/...`
- 命名：`test_*.py` 或 `*_test.py`
- 异步测试自动支持

## 项目特定约定

### 环境变量命名
- 提供商 API key: `{PROVIDER}_API_KEY`
- Base URL: `{PROVIDER}_BASE_URL`
- 示例：`OPENAI_API_KEY`, `SILICONFLOW_BASE_URL`

### 模型名称格式
- 标准格式：`"provider:model-name"`
- 自动推断：`"gpt-4"` → openai
- 别名支持：`"opr:anthropic/claude-3-haiku"` → openrouter