# 项目结构

## 目录组织

```
langgraph_project_template/
├── pyproject.toml              # uv 项目配置
├── uv.lock                     # 依赖锁文件
├── .env.example                # 环境变量模板
├── mcp.json                    # MCP 服务器配置
├── langgraph.json              # LangGraph 配置
├── README.md / README_zh.md    # 项目文档
├── CLAUDE.md                   # Claude Code 指令
│
├── src/app/
│   ├── app.py                  # FastAPI 主应用入口（待实现）
│   └── common/                 # 共享模块
│       ├── models/             # 模型初始化
│       │   ├── chat_models.py       # 聊天模型工厂（核心）
│       │   ├── embedding_models.py  # 嵌入模型
│       │   ├── document_compressor.py
│       │   ├── siliconflow/         # SiliconFlow rerank
│       │   └── dmxapi/              # DMXAPI rerank
│       │
│       ├── tools/              # 工具集成
│       │   ├── mcp.py               # MCP 工具适配
│       │   └── web_search/          # Web 搜索工具
│       │
│       ├── prebuild/           # 预构建组件
│       │   ├── agents/              # 智能体实现
│       │   │   ├── deepagents/      # Deep agents
│       │   │   └── react/           # ReAct agents
│       │   ├── knowledge_builder/   # 知识构建
│       │   ├── knowledge_retriever/ # 知识检索
│       │   └── workflow/            # 工作流
│       │
│       ├── context/            # 上下文管理
│       │   └── middleware/     # 中间件
│       │
│       ├── utils/              # 工具函数
│       │   ├── core.py
│       │   └── load.py
│       │
│       └── text_spiltters/     # 文本分割器
│
├── tests/                      # 测试代码
│   ├── app/                    # 应用测试
│   └── tests.ipynb             # Jupyter 测试笔记本
│
├── examples/                   # 示例代码
│   └── graph/                  # LangGraph 示例
│       ├── react.py
│       └── research_agents.py
│
└── eval/                       # 评估相关

## 架构设计原则

### 1. src/ 布局
- 使用 `src/app/` 布局避免导入问题
- 包名为 `app`，模块结构清晰

### 2. 模块职责分层
- **models/**: 模型初始化和配置
- **tools/**: 外部工具集成
- **prebuild/**: 预构建的可复用组件
- **utils/**: 通用工具函数
- **context/**: 上下文和中间件管理

### 3. 配置驱动
- 使用 `.env` 管理环境变量
- 使用配置类（ProviderConfig）管理提供商
- 工厂模式统一模型创建

### 4. 可扩展性
- 新增提供商只需添加配置
- 模块化设计，易于扩展
- 预留数据库、缓存、存储接口