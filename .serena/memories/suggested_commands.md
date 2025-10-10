# 开发命令速查

## 项目设置

### 初始化环境
```bash
# 克隆项目后首次设置
uv sync                    # 安装所有依赖（包括开发依赖）
cp .env.example .env      # 创建环境变量配置
# 编辑 .env 文件，填入你的 API keys
```

### 依赖管理
```bash
# 添加生产依赖
uv add package-name

# 添加开发依赖
uv add --dev package-name

# 同步依赖（lockfile 变更后）
uv sync

# 更新依赖
uv lock --upgrade
```

## 开发工作流

### 代码质量检查
```bash
# Linting（检查代码问题）
ruff check                 # 检查所有问题
ruff check --fix          # 自动修复可修复的问题

# Formatting（格式化代码）
ruff format               # 格式化所有代码

# 完整质量检查流程
ruff check --fix && ruff format
```

### 测试
```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/app/test_*.py

# 运行匹配模式的测试
pytest -k "test_openai"

# 详细输出
pytest -v

# 覆盖率报告
pytest --cov=src/app
```

### LangGraph 开发
```bash
# 启动 LangGraph 开发服务器
langgraph dev

# 构建 LangGraph
langgraph build

# 测试 graph
langgraph test
```

### 运行应用
```bash
# 开发模式（热重载）
uvicorn app.app:app --reload

# 使用 Gunicorn（生产）
gunicorn app.app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

### Jupyter 开发
```bash
# 启动 Jupyter
jupyter notebook

# 或使用 JupyterLab
jupyter lab
```

## 任务完成检查清单

每次完成开发任务后，运行以下命令确保代码质量：

```bash
# 1. 代码检查和格式化
ruff check --fix && ruff format

# 2. 运行测试
pytest

# 3. 提交前确认
git status
git diff
```

## Darwin (macOS) 特定命令

### 文件操作
```bash
# 列出文件
ls -la                    # 详细列表
ls -lah                   # 人类可读大小

# 查找文件
find . -name "*.py"      # 查找 Python 文件
mdfind "kind:source"     # Spotlight 搜索源代码

# 文件搜索
grep -r "pattern" src/   # 递归搜索
rg "pattern"             # 使用 ripgrep（更快）
```

### 进程管理
```bash
# 查看端口占用
lsof -i :8000            # 查看 8000 端口

# 杀死进程
kill -9 <PID>
```

### Git 操作
```bash
# 查看状态
git status

# 创建分支
git checkout -b feature/name

# 提交
git add .
git commit -m "feat: description"

# 推送
git push origin feature/name
```

## 环境变量配置要点

### 必需配置
- `LANGSMITH_API_KEY`: LangSmith 追踪（可选但推荐）
- 至少一个 LLM 提供商的 API key（OPENAI_API_KEY 等）

### 可选配置
- 数据库连接（Postgres, Redis）
- 向量数据库（Milvus）
- 存储（OpenDAL）
- 搜索工具（TAVILY_API_KEY）

### 国内加速
- 项目已配置清华、阿里云、中科大镜像源
- SiliconFlow、Dashscope 等国内提供商可用