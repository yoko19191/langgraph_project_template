"""Example usage of embedding models with caching.

This demonstrates how to use the enhanced init_embedding_model function with:
- Different embedding providers
- File-based caching for persistent storage
- In-memory caching for temporary storage
- Performance comparisons with and without caching
"""

import os
import sys
import time
from pathlib import Path

# Add project root to Python path for direct execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.app.common.models.embedding_models import init_embedding_model
from src.app.common.utils.logging import get_logger, setup_logging

# Setup logging to see cache operations
setup_logging(log_level="INFO")
logger = get_logger(__name__)


def example_basic_usage():
    """Demonstrate basic embedding without caching."""
    print("\n=== Basic Usage Example (No Caching) ===\n")  # noqa: T201
    
    try:
        # Basic embedding model without caching
        embeddings = init_embedding_model(
            model="text-embedding-3-small",
            provider="openai"
        )
        
        # Generate embeddings for sample texts
        texts = [
            "LangChain is a framework for developing applications powered by language models.",
            "Python is a high-level programming language.",
            "Machine learning enables computers to learn without explicit programming."
        ]
        
        logger.info("Generating embeddings without caching...")
        start_time = time.time()
        
        # This will call the API each time
        vectors = embeddings.embed_documents(texts)
        
        end_time = time.time()
        print(f"Generated {len(vectors)} embeddings in {end_time - start_time:.2f} seconds")  # noqa: T201
        print(f"First embedding dimension: {len(vectors[0])}")  # noqa: T201
        
    except Exception as e:
        print(f"Basic usage failed: {e}")  # noqa: T201
        print("Note: This requires a valid OPENAI_API_KEY environment variable")  # noqa: T201


def example_file_based_caching():
    """Demonstrate file-based caching for persistent storage."""
    print("\n=== File-Based Caching Example ===\n")  # noqa: T201
    
    try:
        # Create embedding model with file-based caching
        embeddings = init_embedding_model(
            model="text-embedding-3-small",
            provider="openai",
            cache=True,
            cache_dir="./cache/embeddings",
            cache_namespace="example"
        )
        
        texts = [
            "LangChain is a framework for developing applications powered by language models.",
            "Python is a high-level programming language.",
            "Machine learning enables computers to learn without explicit programming."
        ]
        
        # First run - will cache embeddings
        logger.info("First run - generating and caching embeddings...")
        start_time = time.time()
        
        vectors1 = embeddings.embed_documents(texts)
        
        end_time = time.time()
        first_run_time = end_time - start_time
        print(f"First run: {len(vectors1)} embeddings in {first_run_time:.2f} seconds")  # noqa: T201
        
        # Second run - should use cached embeddings
        logger.info("Second run - using cached embeddings...")
        start_time = time.time()
        
        vectors2 = embeddings.embed_documents(texts)
        
        end_time = time.time()
        second_run_time = end_time - start_time
        print(f"Second run: {len(vectors2)} embeddings in {second_run_time:.2f} seconds")  # noqa: T201
        
        # Verify embeddings are identical
        vectors_match = all(v1 == v2 for v1, v2 in zip(vectors1, vectors2))
        print(f"Cached embeddings match original: {vectors_match}")  # noqa: T201
        
        if first_run_time > 0:
            speedup = first_run_time / second_run_time if second_run_time > 0 else float('inf')
            print(f"Cache speedup: {speedup:.1f}x faster")  # noqa: T201
        
        # Show cache directory contents
        cache_dir = Path("./cache/embeddings")
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*"))
            print(f"Cache directory contains {len(cache_files)} files")  # noqa: T201
        
    except Exception as e:
        print(f"File-based caching failed: {e}")  # noqa: T201
        print("Note: This requires a valid OPENAI_API_KEY environment variable")  # noqa: T201


def example_in_memory_caching():
    """Demonstrate in-memory caching for temporary storage."""
    print("\n=== In-Memory Caching Example ===\n")  # noqa: T201
    
    try:
        # Create embedding model with in-memory caching
        embeddings = init_embedding_model(
            model="text-embedding-3-small", 
            provider="openai",
            cache=True,
            cache_in_memory=True,
            cache_namespace="memory_example"
        )
        
        texts = [
            "Artificial intelligence is transforming industries.",
            "Deep learning is a subset of machine learning.",
            "Natural language processing enables human-computer interaction."
        ]
        
        # First embedding - will be cached in memory
        logger.info("First embedding - caching in memory...")
        start_time = time.time()
        
        vectors1 = embeddings.embed_documents(texts)
        
        end_time = time.time()
        first_time = end_time - start_time
        print(f"First run: {len(vectors1)} embeddings in {first_time:.2f} seconds")  # noqa: T201
        
        # Second embedding - should use memory cache
        logger.info("Second embedding - using memory cache...")
        start_time = time.time()
        
        vectors2 = embeddings.embed_documents(texts)
        
        end_time = time.time()
        second_time = end_time - start_time
        print(f"Second run: {len(vectors2)} embeddings in {second_time:.2f} seconds")  # noqa: T201
        
        print("Note: In-memory cache is lost when process ends")  # noqa: T201
        
    except Exception as e:
        print(f"In-memory caching failed: {e}")  # noqa: T201
        print("Note: This requires a valid OPENAI_API_KEY environment variable")  # noqa: T201


def example_different_providers():
    """Demonstrate caching with different embedding providers."""
    print("\n=== Different Providers with Caching ===\n")  # noqa: T201
    
    providers_to_test = [
        ("openai", "text-embedding-3-small", "OPENAI_API_KEY"),
        ("dashscope", "text-embedding-v4", "DASHSCOPE_API_KEY"),
        ("siliconflow", "BAAI/bge-small-en-v1.5", "SILICONFLOW_API_KEY"),
    ]
    
    for provider, model, api_key_env in providers_to_test:
        if not os.getenv(api_key_env):
            print(f"Skipping {provider} - no {api_key_env} found")  # noqa: T201
            continue
            
        try:
            print(f"\nTesting {provider} with model {model}")  # noqa: T201
            
            embeddings = init_embedding_model(
                model=model,
                provider=provider,
                cache=True,
                cache_dir=f"./cache/{provider}",
                cache_namespace=f"{provider}_{model}"
            )
            
            # Test with a single text
            test_text = "LangChain enables building LLM applications."
            vector = embeddings.embed_query(test_text)
            
            print(f"✅ {provider}: Generated embedding with {len(vector)} dimensions")  # noqa: T201
            
        except Exception as e:
            print(f"❌ {provider} failed: {e}")  # noqa: T201


def example_cache_namespaces():
    """Demonstrate using different cache namespaces."""
    print("\n=== Cache Namespaces Example ===\n")  # noqa: T201
    
    try:
        # Create two embeddings with different namespaces
        embeddings_v1 = init_embedding_model(
            model="text-embedding-3-small",
            provider="openai",
            cache=True,
            cache_namespace="version_1"
        )
        
        embeddings_v2 = init_embedding_model(
            model="text-embedding-3-small", 
            provider="openai",
            cache=True,
            cache_namespace="version_2"
        )
        
        text = "Cache namespaces allow separate caching contexts."
        
        # Each will have its own cache space
        vector1 = embeddings_v1.embed_query(text)
        vector2 = embeddings_v2.embed_query(text)
        
        print(f"Vector 1 dimensions: {len(vector1)}")  # noqa: T201
        print(f"Vector 2 dimensions: {len(vector2)}")  # noqa: T201
        print("Both vectors are cached separately by namespace")  # noqa: T201
        
    except Exception as e:
        print(f"Cache namespaces example failed: {e}")  # noqa: T201


def cleanup_cache_files():
    """Clean up cache files created during examples."""
    print("\n=== Cleaning Up Cache Files ===\n")  # noqa: T201
    
    cache_dirs = [
        "./cache",
        "./embeddings_cache"
    ]
    
    import shutil
    
    for cache_dir in cache_dirs:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            try:
                shutil.rmtree(cache_path)
                print(f"Removed cache directory: {cache_dir}")  # noqa: T201
            except Exception as e:
                print(f"Failed to remove {cache_dir}: {e}")  # noqa: T201


if __name__ == "__main__":
    print("=== Embedding Models with Caching Examples ===")  # noqa: T201
    print()  # noqa: T201
    print("Note: These examples require API keys for the respective providers.")  # noqa: T201
    print("Set environment variables like OPENAI_API_KEY, DASHSCOPE_API_KEY, etc.")  # noqa: T201
    
    # Run examples
    example_basic_usage()
    example_file_based_caching()
    example_in_memory_caching()
    example_different_providers()
    example_cache_namespaces()
    
    print("\n=== Examples Complete ===\n")  # noqa: T201
    
    # Ask user if they want to clean up cache files
    try:
        response = input("Clean up cache files? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            cleanup_cache_files()
    except (EOFError, KeyboardInterrupt):
        print("\nSkipping cleanup.")  # noqa: T201