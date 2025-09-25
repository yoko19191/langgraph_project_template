"""聊天模型测试套件.

全面测试重构后的聊天模型初始化功能，体现 Python 之禅：
- 明确胜过隐晦：每个测试都有清晰的意图
- 简单胜过复杂：测试逻辑直观易懂
- 错误永远不应该被静默忽略：全面的错误处理测试
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pytest
from langchain_core.language_models import BaseChatModel

from app.common.models.chat_models import (
    init_chat_model,
    _parse_model,
    _normalize_provider_name,
    _attempt_infer_model_provider,
    PROVIDERS
)


class TestModelInitialization:
    """测试模型初始化功能."""

    def test_openai_model(self):
        """测试 OpenAI 模型初始化 - 回退到 langchain 默认实现."""
        model = init_chat_model("gpt-4.1", model_provider="openai")
        assert isinstance(model, BaseChatModel)
        assert hasattr(model, 'model_name')

    def test_anthropic_model(self):
        """测试 Anthropic 模型初始化 - 回退到 langchain 默认实现."""
        model = init_chat_model("claude-sonnet-4-20250514", model_provider="anthropic")
        assert isinstance(model, BaseChatModel)

    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="需要 Groq API 密钥")
    def test_groq_model(self):
        """测试 Groq 模型初始化 - 回退到 langchain 默认实现."""
        model = init_chat_model("openai/gpt-oss-20b", model_provider="groq")
        assert isinstance(model, BaseChatModel)

    def test_deepseek_model(self):
        """测试 DeepSeek 模型初始化 - 回退到 langchain 默认实现."""
        model = init_chat_model("deepseek-chat", model_provider="deepseek")
        assert isinstance(model, BaseChatModel)


class TestCustomProviders:
    """测试自定义提供商功能."""

    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="需要 OpenRouter API 密钥")
    def test_openrouter_model(self):
        """测试 OpenRouter 模型初始化."""
        model = init_chat_model("anthropic/claude-4-sonnet", model_provider="openrouter")
        assert isinstance(model, BaseChatModel)
        assert hasattr(model, 'openai_api_base')
        assert "openrouter.ai" in model.openai_api_base

    @pytest.mark.skipif(not os.getenv("DASHSCOPE_API_KEY"), reason="需要 DashScope API 密钥")
    def test_qwen_model(self):
        """测试通义千问模型初始化."""
        model = init_chat_model("qwen-max", model_provider="dashscope")
        assert isinstance(model, BaseChatModel)

    @pytest.mark.skipif(not os.getenv("DASHSCOPE_API_KEY"), reason="需要 DashScope API 密钥")
    def test_qwq_model(self):
        """测试 QwQ 推理模型 - 应该使用 ChatQwQ 类."""
        model = init_chat_model("qwq-32b-preview", model_provider="dashscope")
        assert isinstance(model, BaseChatModel)
        # QwQ 模型使用特殊的类

    @pytest.mark.skipif(not os.getenv("ZHIPUAI_API_KEY"), reason="需要智谱 API 密钥")
    def test_zhipuai_model(self):
        """测试智谱 GLM 模型初始化."""
        model = init_chat_model("glm-4.5", model_provider="zhipuai")
        assert isinstance(model, BaseChatModel)

    @pytest.mark.skipif(not os.getenv("MOONSHOT_API_KEY"), reason="需要月之暗面 API 密钥")
    def test_moonshot_model(self):
        """测试月之暗面 Kimi 模型初始化."""
        model = init_chat_model("kimi-k2-turbo-preview", model_provider="moonshot")
        assert isinstance(model, BaseChatModel)

    @pytest.mark.skipif(not os.getenv("ARK_API_KEY"), reason="需要火山方舟 API 密钥")
    def test_volcengine_model(self):
        """测试火山方舟豆包模型初始化."""
        model = init_chat_model("doubao-seed-1-6-250615", model_provider="volcengine")
        assert isinstance(model, BaseChatModel)

    @pytest.mark.skipif(not os.getenv("SILICONFLOW_API_KEY"), reason="需要硅基流动 API 密钥")
    def test_siliconflow_model(self):
        """测试硅基流动模型初始化."""
        model = init_chat_model("deepseek-ai/DeepSeek-V3", model_provider="siliconflow")
        assert isinstance(model, BaseChatModel)

    @pytest.mark.skipif(not os.getenv("DMXAPI_API_KEY"), reason="需要 DMXAPI 密钥")
    def test_dmxapi_model(self):
        """测试 DMXAPI 模型初始化."""
        model = init_chat_model("Qwen/Qwen2.5-7B-Instruct", model_provider="dmxapi")
        assert isinstance(model, BaseChatModel)

    @pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="需要 Google API 密钥")
    def test_google_model(self):
        """测试 Google Gemini 模型初始化."""
        model = init_chat_model("gemini-2.5-flash", model_provider="google")
        assert isinstance(model, BaseChatModel)


class TestModelParsing:
    """测试模型解析功能."""

    def test_parse_provider_model_format(self):
        """测试 'provider:model' 格式解析."""
        model, provider = _parse_model("openrouter:anthropic/claude-3-haiku", None)
        assert model == "anthropic/claude-3-haiku"
        assert provider == "openrouter"

    def test_parse_with_explicit_provider(self):
        """测试显式指定提供商."""
        model, provider = _parse_model("gpt-4", "openai")
        assert model == "gpt-4"
        assert provider == "openai"

    def test_parse_inferred_provider(self):
        """测试自动推断提供商."""
        model, provider = _parse_model("qwen-max", None)
        assert model == "qwen-max"
        assert provider == "dashscope"

        model, provider = _parse_model("glm-4", None)
        assert model == "glm-4"
        assert provider == "zhipuai"

        model, provider = _parse_model("kimi-chat", None)
        assert model == "kimi-chat"
        assert provider == "moonshot"

    def test_parse_empty_model(self):
        """测试空模型名称错误处理."""
        with pytest.raises(ValueError, match="模型名称不能为空"):
            _parse_model(None, "openai")

        with pytest.raises(ValueError, match="模型名称不能为空"):
            _parse_model("", "openai")

    def test_parse_unknown_provider(self):
        """测试未知提供商错误处理."""
        with pytest.raises(ValueError, match="无法为模型.*推断提供商"):
            _parse_model("unknown-model-xyz", None)


class TestProviderNormalization:
    """测试提供商名称标准化功能."""

    def test_alias_mapping(self):
        """测试提供商别名映射."""
        assert _normalize_provider_name("opr") == "openrouter"
        assert _normalize_provider_name("qwen") == "dashscope"
        assert _normalize_provider_name("ark") == "volcengine"

    def test_case_normalization(self):
        """测试大小写标准化."""
        assert _normalize_provider_name("OpenRouter") == "openrouter"
        assert _normalize_provider_name("ZHIPUAI") == "zhipuai"

    def test_hyphen_to_underscore(self):
        """测试连字符转下划线."""
        assert _normalize_provider_name("custom-provider") == "custom_provider"


class TestProviderInference:
    """测试提供商推断功能."""

    def test_qwen_models(self):
        """测试通义千问系列模型推断."""
        assert _attempt_infer_model_provider("qwen-max") == "dashscope"
        assert _attempt_infer_model_provider("qwq-32b-preview") == "dashscope"
        assert _attempt_infer_model_provider("qvq-72b-preview") == "dashscope"

    def test_moonshot_models(self):
        """测试月之暗面系列模型推断."""
        assert _attempt_infer_model_provider("moonshot-v1-8k") == "moonshot"
        assert _attempt_infer_model_provider("kimi-chat") == "moonshot"

    def test_volcengine_models(self):
        """测试火山方舟系列模型推断."""
        assert _attempt_infer_model_provider("doubao-lite-4k") == "volcengine"

    def test_zhipuai_models(self):
        """测试智谱系列模型推断."""
        assert _attempt_infer_model_provider("glm-4") == "zhipuai"

    def test_google_models(self):
        """测试 Google 系列模型推断."""
        assert _attempt_infer_model_provider("gemini-1.5-flash") == "google"
        assert _attempt_infer_model_provider("gemma-7b-it") == "google"


class TestConfigurableModels:
    """测试可配置模型功能."""

    def test_configurable_model_creation(self):
        """测试创建可配置模型."""
        configurable_model = init_chat_model(
            model="gpt-4",
            configurable_fields=("model", "temperature")
        )
        # 可配置模型应该不是直接的 BaseChatModel 实例
        assert hasattr(configurable_model, 'configurable_fields')

    def test_default_configurable_fields(self):
        """测试默认可配置字段."""
        configurable_model = init_chat_model()  # 无参数调用
        assert hasattr(configurable_model, 'configurable_fields')


class TestProviderConfiguration:
    """测试提供商配置系统."""

    def test_all_providers_configured(self):
        """测试所有提供商都有正确配置."""
        expected_providers = [
            "openrouter", "dashscope", "zhipuai", "moonshot",
            "volcengine", "siliconflow", "dmxapi", "xinference", "google"
        ]

        for provider in expected_providers:
            assert provider in PROVIDERS
            config = PROVIDERS[provider]
            assert config.package
            assert config.module
            assert config.class_name

    def test_provider_config_structure(self):
        """测试提供商配置结构完整性."""
        for provider, config in PROVIDERS.items():
            # 基本字段必须存在
            assert isinstance(config.package, str)
            assert isinstance(config.module, str)
            assert isinstance(config.class_name, str)

            # 可选字段类型检查
            if config.api_key_env is not None:
                assert isinstance(config.api_key_env, str)
            if config.base_url_env is not None:
                assert isinstance(config.base_url_env, str)


# 运行测试的便捷函数
def run_basic_tests():
    """运行基础功能测试 - 适合快速验证."""
    print("🧪 开始基础功能测试...")

    # 测试模型解析
    try:
        model, provider = _parse_model("qwen-max", None)
        print(f"✅ 模型解析测试通过: {model} -> {provider}")
    except Exception as e:
        print(f"❌ 模型解析测试失败: {e}")

    # 测试提供商标准化
    try:
        normalized = _normalize_provider_name("opr")
        assert normalized == "openrouter"
        print(f"✅ 提供商标准化测试通过: opr -> {normalized}")
    except Exception as e:
        print(f"❌ 提供商标准化测试失败: {e}")

    # 测试配置系统
    try:
        assert "openrouter" in PROVIDERS
        assert PROVIDERS["openrouter"].class_name == "ChatOpenAI"
        print("✅ 配置系统测试通过")
    except Exception as e:
        print(f"❌ 配置系统测试失败: {e}")

    print("🎉 基础功能测试完成!")


if __name__ == "__main__":
    # 直接运行时执行基础测试
    run_basic_tests()