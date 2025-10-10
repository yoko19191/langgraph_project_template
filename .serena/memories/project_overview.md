# 项目概览

## 项目目的
这是一个**生产级 FastAPI + LangGraph 模板**，专为构建可扩展的智能体应用而设计。项目强调可扩展性和 Pythonic 设计原则，提供：

- 多 AI 模型提供商支持（9+ 国际和国内提供商）
- MCP (Model Context Protocol) 集成
- 配置驱动的架构设计
- 完整的代码质量工具链

## 技术栈

### 核心框架
- **Python**: 3.12+ (使用现代类型提示和特性)
- **FastAPI**: Web 框架
- **LangGraph**: 智能体编排框架
- **LangChain**: LLM 应用开发框架

### 包管理
- **uv**: 现代 Python 包管理器（替代 pip/poetry）
- 使用中国镜像源加速下载

### 主要依赖
- **LLM Providers**: langchain-openai, langchain-anthropic, langchain-google-genai, langchain-groq, langchain-deepseek 等
- **中国提供商**: dashscope (Qwen), langchain-siliconflow 等
- **工具**: tavily-python (搜索), wikipedia-api, arxiv
- **向量存储**: langchain-milvus, faiss-cpu
- **异步**: aiofiles, celery
- **Web**: fastapi, uvicorn, gunicorn

### 开发工具
- **测试**: pytest (async 支持)
- **代码质量**: ruff (linter + formatter)
- **开发**: jupyter, langgraph-cli

## 项目哲学
遵循 Python 之禅和 Guido 的设计原则：
- 可读性至上
- 明确胜过隐晦
- 简单胜过复杂
- 配置驱动的设计
- 类型安全和文档完善