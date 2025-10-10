"""Tests for embedding model initialization with caching support."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain.embeddings.base import Embeddings

from src.app.common.models.embedding_models import (
    EMBEDDING_PROVIDERS,
    EmbeddingProviderConfig,
    _build_embedding_params,
    _create_cached_embeddings,
    _create_embedding_model,
    init_embedding_model,
)


class MockEmbeddings(Embeddings):
    """Mock embedding class for testing."""
    
    def __init__(self, model: str = "mock-model", **kwargs):
        self.model = model
        self.kwargs = kwargs
    
    def embed_documents(self, texts):
        """Return mock embeddings for documents."""
        return [[0.1, 0.2, 0.3] for _ in texts]
    
    def embed_query(self, text):
        """Return mock embedding for query."""
        return [0.1, 0.2, 0.3]


class TestEmbeddingProviderConfig:
    """Test EmbeddingProviderConfig dataclass."""
    
    def test_provider_config_creation(self):
        """Test creating provider configuration."""
        config = EmbeddingProviderConfig(
            package="test-package",
            module="test.module",
            class_name="TestEmbeddings",
            api_key_env="TEST_API_KEY"
        )
        
        assert config.package == "test-package"
        assert config.module == "test.module"
        assert config.class_name == "TestEmbeddings"
        assert config.api_key_env == "TEST_API_KEY"
        assert config.base_url_env is None
        assert config.default_base_url is None
        assert config.special_params is None


class TestInitEmbeddingModel:
    """Test init_embedding_model function."""
    
    def test_provider_alias_handling(self):
        """Test provider alias parameter handling."""
        with patch('src.app.common.models.embedding_models._create_embedding_model') as mock_create:
            mock_create.return_value = MockEmbeddings()
            
            # Test using provider alias
            init_embedding_model(model="test", provider="openai")
            mock_create.assert_called_once_with(model="test", provider="openai")
    
    def test_provider_conflict_error(self):
        """Test error when both provider and model_provider are specified."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            init_embedding_model(
                model="test",
                provider="openai", 
                model_provider="anthropic"
            )
    
    def test_missing_provider_error(self):
        """Test error when no provider is specified."""
        with pytest.raises(ValueError, match="model_provider.*must be specified"):
            init_embedding_model(model="test")
    
    def test_provider_normalization(self):
        """Test provider name normalization."""
        with patch('src.app.common.models.embedding_models._create_embedding_model') as mock_create:
            mock_create.return_value = MockEmbeddings()
            
            # Test hyphen to underscore conversion
            init_embedding_model(model="test", provider="open-ai")
            mock_create.assert_called_once_with(model="test", provider="open_ai")
    
    def test_basic_embedding_without_cache(self):
        """Test basic embedding model creation without caching."""
        with patch('src.app.common.models.embedding_models._create_embedding_model') as mock_create:
            mock_embeddings = MockEmbeddings()
            mock_create.return_value = mock_embeddings
            
            result = init_embedding_model(
                model="test-model",
                provider="openai",
                cache=False
            )
            
            assert result is mock_embeddings
            mock_create.assert_called_once_with(model="test-model", provider="openai")
    
    def test_embedding_with_cache_enabled(self):
        """Test embedding model creation with caching enabled."""
        with patch('src.app.common.models.embedding_models._create_embedding_model') as mock_create, \
             patch('src.app.common.models.embedding_models._create_cached_embeddings') as mock_cache:
            
            mock_base_embeddings = MockEmbeddings()
            mock_cached_embeddings = MockEmbeddings()
            mock_create.return_value = mock_base_embeddings
            mock_cache.return_value = mock_cached_embeddings
            
            result = init_embedding_model(
                model="test-model",
                provider="openai",
                cache=True,
                cache_dir="./test_cache"
            )
            
            assert result is mock_cached_embeddings
            mock_create.assert_called_once_with(model="test-model", provider="openai")
            mock_cache.assert_called_once_with(
                base_embeddings=mock_base_embeddings,
                model="test-model",
                cache_dir="./test_cache",
                cache_namespace=None,
                cache_in_memory=False
            )


class TestCreateCachedEmbeddings:
    """Test _create_cached_embeddings function."""
    
    def test_file_based_caching(self):
        """Test file-based caching setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('langchain.embeddings.CacheBackedEmbeddings') as mock_cache_class, \
                 patch('langchain.storage.LocalFileStore') as mock_store_class, \
                 patch('langchain.storage.InMemoryByteStore') as mock_memory_store_class:
                
                mock_store = Mock()
                mock_store_class.return_value = mock_store
                
                mock_cached_embeddings = Mock()
                mock_cache_class.from_bytes_store.return_value = mock_cached_embeddings
                
                base_embeddings = MockEmbeddings(model="test-model")
                
                result = _create_cached_embeddings(
                    base_embeddings=base_embeddings,
                    model="test-model",
                    cache_dir=temp_dir,
                    cache_namespace="test-namespace",
                    cache_in_memory=False
                )
                
                assert result is mock_cached_embeddings
                mock_store_class.assert_called_once_with(temp_dir)
                mock_cache_class.from_bytes_store.assert_called_once_with(
                    underlying_embeddings=base_embeddings,
                    document_embedding_store=mock_store,
                    namespace="test-namespace"
                )
    
    def test_in_memory_caching(self):
        """Test in-memory caching setup."""
        with patch('langchain.embeddings.CacheBackedEmbeddings') as mock_cache_class, \
             patch('langchain.storage.InMemoryByteStore') as mock_store_class:
            
            mock_store = Mock()
            mock_store_class.return_value = mock_store
            
            mock_cached_embeddings = Mock()
            mock_cache_class.from_bytes_store.return_value = mock_cached_embeddings
            
            base_embeddings = MockEmbeddings(model="test-model")
            
            result = _create_cached_embeddings(
                base_embeddings=base_embeddings,
                model="test-model",
                cache_dir=None,
                cache_namespace="test-namespace",
                cache_in_memory=True
            )
            
            assert result is mock_cached_embeddings
            mock_store_class.assert_called_once()
            mock_cache_class.from_bytes_store.assert_called_once_with(
                underlying_embeddings=base_embeddings,
                document_embedding_store=mock_store,
                namespace="test-namespace"
            )
    
    def test_cache_namespace_from_model_attribute(self):
        """Test namespace derivation from embedding model attribute."""
        with patch('langchain.embeddings.CacheBackedEmbeddings') as mock_cache_class, \
             patch('langchain.storage.InMemoryByteStore'):
            
            mock_cached_embeddings = Mock()
            mock_cache_class.from_bytes_store.return_value = mock_cached_embeddings
            
            base_embeddings = MockEmbeddings(model="test-model-123")
            
            _create_cached_embeddings(
                base_embeddings=base_embeddings,
                model=None,
                cache_dir=None,
                cache_namespace=None,  # Should derive from base_embeddings.model
                cache_in_memory=True
            )
            
            # Check that namespace was derived from model attribute
            call_args = mock_cache_class.from_bytes_store.call_args
            assert call_args[1]["namespace"] == "test-model-123"
    
    def test_cache_namespace_from_model_param(self):
        """Test namespace derivation from model parameter."""
        with patch('langchain.embeddings.CacheBackedEmbeddings') as mock_cache_class, \
             patch('langchain.storage.InMemoryByteStore'):
            
            mock_cached_embeddings = Mock()
            mock_cache_class.from_bytes_store.return_value = mock_cached_embeddings
            
            # Mock embeddings without model attribute
            base_embeddings = Mock(spec=Embeddings)
            delattr(base_embeddings, 'model')  # Ensure no model attribute
            
            _create_cached_embeddings(
                base_embeddings=base_embeddings,
                model="param-model",
                cache_dir=None,
                cache_namespace=None,  # Should derive from model parameter
                cache_in_memory=True
            )
            
            # Check that namespace was derived from model parameter
            call_args = mock_cache_class.from_bytes_store.call_args
            assert call_args[1]["namespace"] == "param-model"
    
    def test_cache_namespace_from_class_name(self):
        """Test namespace derivation from class name as fallback."""
        with patch('langchain.embeddings.CacheBackedEmbeddings') as mock_cache_class, \
             patch('langchain.storage.InMemoryByteStore'):
            
            mock_cached_embeddings = Mock()
            mock_cache_class.from_bytes_store.return_value = mock_cached_embeddings
            
            # Mock embeddings without model attribute
            base_embeddings = Mock(spec=Embeddings)
            delattr(base_embeddings, 'model')  # Ensure no model attribute
            base_embeddings.__class__.__name__ = "CustomEmbeddings"
            
            _create_cached_embeddings(
                base_embeddings=base_embeddings,
                model=None,  # No model parameter
                cache_dir=None,
                cache_namespace=None,  # Should derive from class name
                cache_in_memory=True
            )
            
            # Check that namespace was derived from class name
            call_args = mock_cache_class.from_bytes_store.call_args
            assert call_args[1]["namespace"] == "CustomEmbeddings"
    
    def test_cache_directory_creation(self):
        """Test that cache directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "new_cache_dir"
            
            with patch('langchain.embeddings.CacheBackedEmbeddings') as mock_cache_class, \
                 patch('langchain.storage.LocalFileStore') as mock_store_class:
                
                mock_cache_class.from_bytes_store.return_value = Mock()
                
                base_embeddings = MockEmbeddings()
                
                _create_cached_embeddings(
                    base_embeddings=base_embeddings,
                    model="test",
                    cache_dir=str(cache_path),
                    cache_namespace="test",
                    cache_in_memory=False
                )
                
                # Check that directory was created
                assert cache_path.exists()
                assert cache_path.is_dir()
    
    def test_file_cache_fallback_to_memory(self):
        """Test fallback to in-memory cache when LocalFileStore is not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('langchain.embeddings.CacheBackedEmbeddings') as mock_cache_class, \
                 patch('langchain.storage.InMemoryByteStore') as mock_memory_store_class:
                
                # Mock LocalFileStore to raise ImportError
                with patch('langchain.storage.LocalFileStore', side_effect=ImportError("LocalFileStore not available")):
                    mock_memory_store = Mock()
                    mock_memory_store_class.return_value = mock_memory_store
                    
                    mock_cached_embeddings = Mock()
                    mock_cache_class.from_bytes_store.return_value = mock_cached_embeddings
                    
                    base_embeddings = MockEmbeddings(model="test-model")
                    
                    result = _create_cached_embeddings(
                        base_embeddings=base_embeddings,
                        model="test-model",
                        cache_dir=temp_dir,  # Request file cache
                        cache_namespace="test-namespace",
                        cache_in_memory=False  # Request file cache
                    )
                    
                    # Should fallback to in-memory store
                    assert result is mock_cached_embeddings
                    mock_memory_store_class.assert_called_once()
                    mock_cache_class.from_bytes_store.assert_called_once_with(
                        underlying_embeddings=base_embeddings,
                        document_embedding_store=mock_memory_store,
                        namespace="test-namespace"
                    )
    
    def test_cache_fallback_on_import_error(self):
        """Test fallback to non-cached embeddings when caching imports fail."""
        with patch('builtins.__import__', side_effect=ImportError("Module not found")):
            base_embeddings = MockEmbeddings()
            
            result = _create_cached_embeddings(
                base_embeddings=base_embeddings,
                model="test",
                cache_dir=None,
                cache_namespace=None,
                cache_in_memory=False
            )
            
            # Should return original embeddings when import fails
            assert result is base_embeddings


class TestCreateEmbeddingModel:
    """Test _create_embedding_model function."""
    
    def test_unsupported_provider(self):
        """Test error for unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            _create_embedding_model(model="test", provider="unsupported")
    
    def test_import_error_handling(self):
        """Test handling of missing package imports."""
        # Mock provider that doesn't exist
        test_config = EmbeddingProviderConfig(
            package="nonexistent-package",
            module="nonexistent.module",
            class_name="NonexistentEmbeddings"
        )
        
        with patch.dict(EMBEDDING_PROVIDERS, {"test": test_config}):
            with pytest.raises(ImportError, match="Cannot import module"):
                _create_embedding_model(model="test", provider="test")
    
    def test_class_not_found_error(self):
        """Test handling when embedding class is not found in module."""
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_import.return_value = mock_module
            # Remove the expected class from the module
            delattr(mock_module, 'OpenAIEmbeddings')
            
            with pytest.raises(AttributeError, match="Class.*not found"):
                _create_embedding_model(model="test", provider="openai")


class TestBuildEmbeddingParams:
    """Test _build_embedding_params function."""
    
    def test_basic_params_building(self):
        """Test basic parameter building."""
        config = EmbeddingProviderConfig(
            package="test-package",
            module="test.module", 
            class_name="TestEmbeddings"
        )
        
        with patch('src.app.common.models.embedding_models._get_api_key') as mock_get_key, \
             patch('src.app.common.models.embedding_models._get_env_var') as mock_get_env:
            
            mock_get_key.return_value = None
            mock_get_env.return_value = None
            
            params = _build_embedding_params(
                config=config,
                model="test-model",
                custom_param="custom_value"
            )
            
            expected = {
                "model": "test-model",
                "custom_param": "custom_value"
            }
            assert params == expected
    
    def test_api_key_handling(self):
        """Test API key parameter building."""
        config = EmbeddingProviderConfig(
            package="test-package",
            module="test.module",
            class_name="TestEmbeddings",
            api_key_env="TEST_API_KEY"
        )
        
        with patch('src.app.common.models.embedding_models._get_api_key') as mock_get_key, \
             patch('src.app.common.models.embedding_models._get_env_var') as mock_get_env:
            
            mock_get_key.return_value = "test-key-123"
            mock_get_env.return_value = None
            
            params = _build_embedding_params(config=config, model="test")
            
            assert params["api_key"] == "test-key-123"
    
    def test_google_api_key_special_handling(self):
        """Test special handling for Google API key parameter name."""
        config = EmbeddingProviderConfig(
            package="test-package",
            module="test.module",
            class_name="TestEmbeddings",
            api_key_env="GOOGLE_API_KEY"
        )
        
        with patch('src.app.common.models.embedding_models._get_api_key') as mock_get_key, \
             patch('src.app.common.models.embedding_models._get_env_var') as mock_get_env:
            
            mock_get_key.return_value = "google-key-123"
            mock_get_env.return_value = None
            
            params = _build_embedding_params(config=config, model="test")
            
            # Should use google_api_key instead of api_key
            assert params["google_api_key"] == "google-key-123"
            assert "api_key" not in params
    
    def test_dashscope_api_key_special_handling(self):
        """Test special handling for DashScope API key."""
        config = EmbeddingProviderConfig(
            package="test-package",
            module="test.module",
            class_name="TestEmbeddings",
            api_key_env="DASHSCOPE_API_KEY"
        )
        
        # Mock SecretStr-like object
        mock_secret = Mock()
        mock_secret.get_secret_value.return_value = "dashscope-secret-value"
        
        with patch('src.app.common.models.embedding_models._get_api_key') as mock_get_key, \
             patch('src.app.common.models.embedding_models._get_env_var') as mock_get_env:
            
            mock_get_key.return_value = mock_secret
            mock_get_env.return_value = None
            
            params = _build_embedding_params(config=config, model="test")
            
            # Should extract secret value for DashScope
            assert params["dashscope_api_key"] == "dashscope-secret-value"
    
    def test_base_url_handling(self):
        """Test base URL parameter building."""
        config = EmbeddingProviderConfig(
            package="test-package",
            module="test.module",
            class_name="TestEmbeddings",
            base_url_env="TEST_BASE_URL",
            default_base_url="https://default.example.com"
        )
        
        with patch('src.app.common.models.embedding_models._get_api_key') as mock_get_key, \
             patch('src.app.common.models.embedding_models._get_env_var') as mock_get_env:
            
            mock_get_key.return_value = None
            mock_get_env.return_value = "https://custom.example.com"
            
            params = _build_embedding_params(config=config, model="test")
            
            assert params["base_url"] == "https://custom.example.com"
    
    def test_default_base_url_when_env_not_set(self):
        """Test using default base URL when environment variable is not set."""
        config = EmbeddingProviderConfig(
            package="test-package",
            module="test.module",
            class_name="TestEmbeddings",
            base_url_env="TEST_BASE_URL",
            default_base_url="https://default.example.com"
        )
        
        with patch('src.app.common.models.embedding_models._get_api_key') as mock_get_key, \
             patch('src.app.common.models.embedding_models._get_env_var') as mock_get_env:
            
            mock_get_key.return_value = None
            mock_get_env.return_value = None  # No environment variable set
            
            params = _build_embedding_params(config=config, model="test")
            
            assert params["base_url"] == "https://default.example.com"
    
    def test_xinference_special_handling(self):
        """Test special parameter handling for Xinference."""
        config = EmbeddingProviderConfig(
            package="test-package",
            module="test.module",
            class_name="XinferenceEmbeddings",
            special_params={
                "server_url_env": "XINFERENCE_SERVER_URL",
                "model_uid_env": "XINFERENCE_EMBEDDING_MODEL_UID",
            }
        )
        
        with patch('src.app.common.models.embedding_models._get_api_key') as mock_get_key, \
             patch('src.app.common.models.embedding_models._get_env_var') as mock_get_env:
            
            mock_get_key.return_value = None
            mock_get_env.side_effect = lambda key: {
                "XINFERENCE_SERVER_URL": "http://localhost:9997",
                "XINFERENCE_EMBEDDING_MODEL_UID": "model-123"
            }.get(key)
            
            params = _build_embedding_params(config=config, model="test-model")
            
            # Model should be mapped to model_uid for Xinference
            assert params["model_uid"] == "test-model"
            assert "model" not in params
            # Server URL should be set
            assert params["server_url"] == "http://localhost:9997"


class TestIntegration:
    """Integration tests for embedding model initialization with caching."""
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not available"
    )
    def test_real_openai_embeddings_with_cache(self):
        """Test real OpenAI embeddings with file-based caching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create embeddings with caching
            embeddings = init_embedding_model(
                model="text-embedding-3-small",
                provider="openai", 
                cache=True,
                cache_dir=temp_dir,
                cache_namespace="integration_test"
            )
            
            # Test embedding generation
            test_text = "This is a test for cached embeddings."
            vector = embeddings.embed_query(test_text)
            
            assert isinstance(vector, list)
            assert len(vector) > 0
            assert all(isinstance(x, (int, float)) for x in vector)
            
            # Check that cache files were created
            cache_files = list(Path(temp_dir).glob("*"))
            assert len(cache_files) > 0
    
    def test_mock_embeddings_caching_workflow(self):
        """Test complete caching workflow with mocked embeddings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('src.app.common.models.embedding_models._create_embedding_model') as mock_create:
                mock_embeddings = MockEmbeddings(model="test-model")
                mock_create.return_value = mock_embeddings
                
                # Create cached embeddings
                cached_embeddings = init_embedding_model(
                    model="test-model",
                    provider="openai",
                    cache=True,
                    cache_dir=temp_dir
                )
                
                # Test that caching components were properly initialized
                # This tests the integration without requiring real API calls
                assert hasattr(cached_embeddings, 'embed_query')
                assert hasattr(cached_embeddings, 'embed_documents')
                
                # Test embedding generation
                test_texts = ["Text 1", "Text 2", "Text 3"]
                vectors = cached_embeddings.embed_documents(test_texts)
                
                assert len(vectors) == 3
                assert all(len(v) == 3 for v in vectors)  # Mock returns [0.1, 0.2, 0.3]


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v", "-s"])