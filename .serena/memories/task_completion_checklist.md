# 任务完成检查清单

## 开发任务完成后必做步骤

每次完成开发任务（新功能、bug 修复、重构）后，按以下顺序执行：

### 1. 代码质量检查 ✅

```bash
# 第一步：Linting 和自动修复
ruff check --fix

# 第二步：代码格式化
ruff format

# 组合命令（推荐）
ruff check --fix && ruff format
```

**检查内容**：
- 代码风格（pycodestyle）
- 潜在错误（pyflakes）
- 导入排序（isort）
- 文档字符串（pydocstyle, Google 风格）
- 现代 Python 语法（pyupgrade）

### 2. 运行测试 🧪

```bash
# 运行所有测试
pytest

# 详细输出（推荐）
pytest -v

# 特定模块测试
pytest tests/app/

# 覆盖率检查（可选）
pytest --cov=src/app
```

**确认事项**：
- ✅ 所有测试通过
- ✅ 新功能已添加测试
- ✅ 边界情况已覆盖
- ✅ 异步测试正确执行

### 3. 类型检查（可选但推荐）

```bash
# 如果配置了 mypy
mypy src/app/
```

### 4. 提交前检查 📝

```bash
# 查看修改内容
git status
git diff

# 确认文件状态
git add -A
git status
```

**检查清单**：
- ✅ 没有调试代码（print, pdb）
- ✅ 没有临时文件（.pyc, __pycache__）
- ✅ .env 文件未被提交
- ✅ 测试通过且代码格式化完成

### 5. 提交规范

```bash
# 使用语义化提交信息
git commit -m "feat: add OpenAI provider support"
git commit -m "fix: resolve model inference issue"  
git commit -m "docs: update README with examples"
git commit -m "refactor: simplify provider config"
git commit -m "test: add coverage for edge cases"
```

**提交类型**：
- `feat`: 新功能
- `fix`: bug 修复
- `docs`: 文档更新
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建/工具配置

## 新功能开发检查清单

### 添加新 Provider 时

1. ✅ 在 `PROVIDERS` 字典添加配置
2. ✅ 在 `.env.example` 添加环境变量示例
3. ✅ 更新文档说明支持的提供商
4. ✅ 添加推断逻辑（如果需要）
5. ✅ 编写测试用例
6. ✅ 运行 `ruff check --fix && ruff format`
7. ✅ 运行 `pytest`

### 添加新功能模块时

1. ✅ 遵循 src/app/ 目录结构
2. ✅ 添加 `__init__.py` 导出公共 API
3. ✅ 编写 Google 风格文档字符串
4. ✅ 添加类型注解（Python 3.12+ 语法）
5. ✅ 创建对应测试文件
6. ✅ 更新 README 或相关文档
7. ✅ 运行完整质量检查

## 发布前检查清单

### 版本发布准备

```bash
# 1. 确保所有测试通过
pytest -v

# 2. 代码质量检查
ruff check --fix && ruff format

# 3. 更新版本号
# 编辑 pyproject.toml 中的 version

# 4. 更新 CHANGELOG（如果有）

# 5. 构建检查
uv build

# 6. 标签和发布
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```

## 快速命令参考

### 日常开发循环
```bash
# 编码 → 检查 → 测试 → 提交
ruff check --fix && ruff format && pytest && git status
```

### 问题排查
```bash
# 1. 查看详细测试输出
pytest -vv

# 2. 运行特定失败的测试
pytest tests/app/test_chat_models.py::test_openai -v

# 3. 查看 Ruff 检查详情
ruff check --output-format=full
```

### 清理环境
```bash
# 删除缓存
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# 重新同步依赖
uv sync --reinstall
```

## 常见问题预防

### ❌ 不要做的事
1. ❌ 提交前不运行测试
2. ❌ 忽略 Ruff 警告
3. ❌ 提交包含 print() 的调试代码
4. ❌ 跳过文档字符串编写
5. ❌ 不添加类型注解
6. ❌ 提交 .env 文件

### ✅ 好的实践
1. ✅ 频繁提交，小步迭代
2. ✅ 每个提交信息清晰明确
3. ✅ 保持测试覆盖率
4. ✅ 及时更新文档
5. ✅ 代码审查前自查
6. ✅ 遵循项目规范