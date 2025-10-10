# 核心架构：聊天模型系统

## 架构概览

`src/app/common/models/chat_models.py` 实现了一个**配置驱动的多提供商聊天模型工厂系统**。

### 核心设计模式

1. **工厂模式**：统一的 `init_chat_model()` 入口
2. **配置驱动**：`ProviderConfig` 数据类定义提供商
3. **策略模式**：`PROVIDERS` 字典映射提供商配置
4. **自动推断**：智能推断模型提供商

## 核心组件

### 1. ProviderConfig 数据类
```python
@dataclass
class ProviderConfig:
    package: str              # langchain 包名
    module: str              # 模块路径
    class_name: str          # 模型类名
    api_key_env: str | None  # API key 环境变量
    base_url_env: str | None # Base URL 环境变量
    default_base_url: str | None
    special_params: Dict[str, Any] | None
```

**设计优势**：
- 类型安全（使用 @dataclass）
- 清晰的配置结构
- 易于扩展新提供商

### 2. PROVIDERS 配置字典

支持的提供商（20+ 个）：

**国际提供商**：
- openai, anthropic, google, groq, deepseek
- openrouter (别名: opr)

**中国提供商**：
- dashscope/qwen (阿里云)
- zhipuai (智谱 AI)
- moonshot (月之暗面)
- volcengine/ark (字节跳动)
- siliconflow (硅基流动)
- dmxapi

**本地/自定义**：
- ollama (本地部署)
- xinference (开源模型服务)

### 3. 核心函数流程

```python
init_chat_model()
    ↓
_parse_model()  # 解析 model 和 provider
    ↓
_attempt_infer_model_provider()  # 智能推断
    ↓
_normalize_provider_name()  # 标准化别名
    ↓
_create_chat_model()  # 创建模型实例
    ↓
_get_model_class_name()  # 获取类名（特殊逻辑）
_build_model_params()   # 构建参数
```

## 关键特性

### 1. 智能推断
根据模型名称前缀自动识别提供商：

```python
"gpt-4" → openai
"claude-3" → anthropic  
"qwen-max" → dashscope
"glm-4" → zhipuai
"moonshot-v1" → moonshot
```

### 2. 多种调用方式

```python
# 方式1：自动推断
model = init_chat_model("gpt-4")

# 方式2：显式指定
model = init_chat_model("gpt-4", model_provider="openai")

# 方式3：别名格式
model = init_chat_model("opr:anthropic/claude-3-haiku")

# 方式4：provider:model 格式
model = init_chat_model("openai:gpt-4")
```

### 3. 环境变量驱动

自动从环境变量读取配置：
- API Key: `{PROVIDER}_API_KEY`
- Base URL: `{PROVIDER}_BASE_URL`

例如：
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`
- `SILICONFLOW_API_KEY`, `SILICONFLOW_BASE_URL`

### 4. 特殊逻辑处理

**Dashscope/Qwen 特殊逻辑**：
```python
# QwQ 模型使用 ChatTongyi
if model.startswith("qwq"):
    class_name = "ChatTongyi"
else:
    class_name = config.class_name  # ChatDashscope
```

### 5. 可配置字段支持

```python
# 支持运行时配置
model = init_chat_model(
    configurable_fields=("model", "model_provider"),
    config_prefix="chat"
)
```

## 扩展指南

### 添加新提供商步骤

1. **添加 ProviderConfig**：
```python
"new_provider": ProviderConfig(
    package="langchain-new-provider",
    module="langchain_new_provider",
    class_name="ChatNewProvider",
    api_key_env="NEW_PROVIDER_API_KEY",
    base_url_env="NEW_PROVIDER_BASE_URL",
)
```

2. **添加环境变量**（.env.example）：
```bash
NEW_PROVIDER_API_KEY="your-api-key"
NEW_PROVIDER_BASE_URL="https://api.provider.com/v1"
```

3. **添加推断逻辑**（可选）：
```python
if model.startswith("new-prefix-"):
    return "new_provider"
```

4. **添加别名**（可选）：
```python
_SUPPORTED_PROVIDERS = [..., "new_provider", "np"]
```

5. **测试**：
```python
def test_new_provider():
    model = init_chat_model("new-prefix-model")
    assert isinstance(model, ChatNewProvider)
```

## 最佳实践

### ✅ 推荐做法

1. **使用类型注解**：
```python
def init_chat_model(...) -> BaseChatModel | _ConfigurableModel:
```

2. **清晰的文档字符串**：
```python
"""Initialize chat model with enhanced provider support.

Supported providers:
- Standard: openai, anthropic, ...
- Chinese: dashscope, zhipuai, ...
"""
```

3. **错误处理**：
```python
if not model and not configurable_fields:
    raise ValueError("Model name is required")
```

4. **日志记录**：
```python
logger = logging.getLogger(__name__)
logger.debug(f"Creating {provider} model: {model}")
```

### ❌ 避免做法

1. ❌ 硬编码 API keys
2. ❌ 使用 if-else 链而非字典映射
3. ❌ 忽略类型注解
4. ❌ 缺少文档说明

## 配套模块

### embedding_models.py
- 类似架构，用于嵌入模型
- `EmbeddingProviderConfig` 和 `EMBEDDING_PROVIDERS`
- `init_embedding_model()` 工厂函数

### document_compressor.py
- 文档压缩/重排序
- 集成 rerank 模型

### siliconflow/dmxapi 子包
- 自定义 rerank 实现
- 提供商特定扩展

## 性能考虑

1. **延迟导入**：只在需要时导入提供商包
2. **配置缓存**：PROVIDERS 字典只初始化一次
3. **环境变量读取**：使用 `python-dotenv` 自动加载

## 测试策略

### 单元测试重点
- 模型名称解析（_parse_model）
- 提供商推断（_attempt_infer_model_provider）
- 别名标准化（_normalize_provider_name）
- 参数构建（_build_model_params）

### 集成测试
- 实际创建各提供商模型（需要 API key）
- 测试自动跳过无 key 的提供商