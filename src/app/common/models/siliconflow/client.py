"""SiliconFlow API Client for Rerank."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import httpx


class SiliconFlowClient:
    """SiliconFlow API client supporting both sync and async operations.

    Provides methods to call SiliconFlow's rerank API with full parameter support.

    Args:
        api_key: SiliconFlow API key. If not provided, reads from SILICONFLOW_API_KEY env var.
        base_url: Base URL for SiliconFlow API. Defaults to official endpoint.
        timeout: Request timeout in seconds.

    Example:
        >>> client = SiliconFlowClient()
        >>> result = client.rerank(
        ...     query="Apple",
        ...     documents=["apple", "banana"],
        ...     model="BAAI/bge-reranker-v2-m3"
        ... )
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.siliconflow.cn/v1",
        timeout: float = 30.0,
    ):
        """Initialize SiliconFlow client."""
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SiliconFlow API key must be provided either through "
                "api_key parameter or SILICONFLOW_API_KEY environment variable"
            )

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    @property
    def sync_client(self) -> httpx.Client:
        """Lazy-loaded synchronous HTTP client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._sync_client

    @property
    def async_client(self) -> httpx.AsyncClient:
        """Lazy-loaded asynchronous HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._async_client

    def rerank(
        self,
        query: str,
        documents: List[str],
        model: str = "BAAI/bge-reranker-v2-m3",
        top_n: int | None = None,
        instruction: str | None = None,
        return_documents: bool = False,
        max_chunks_per_doc: int | None = None,
        overlap_tokens: int | None = None,
    ) -> Dict[str, Any]:
        """Rerank documents synchronously using SiliconFlow API.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            model: Model identifier for reranking.
            top_n: Number of top results to return. If None, returns all.
            instruction: Optional instruction to guide the reranking process.
            return_documents: Whether to return document texts in response.
            max_chunks_per_doc: Maximum chunks per document for long documents.
            overlap_tokens: Number of overlapping tokens between chunks.

        Returns:
            API response containing reranked results.

        Raises:
            httpx.HTTPStatusError: If API request fails.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "query": query,
            "documents": documents,
        }

        # Add optional parameters
        if top_n is not None:
            payload["top_n"] = top_n
        if instruction is not None:
            payload["instruction"] = instruction
        if return_documents:
            payload["return_documents"] = return_documents
        if max_chunks_per_doc is not None:
            payload["max_chunks_per_doc"] = max_chunks_per_doc
        if overlap_tokens is not None:
            payload["overlap_tokens"] = overlap_tokens

        response = self.sync_client.post("/rerank", json=payload)
        response.raise_for_status()
        return response.json()

    async def arerank(
        self,
        query: str,
        documents: List[str],
        model: str = "BAAI/bge-reranker-v2-m3",
        top_n: int | None = None,
        instruction: str | None = None,
        return_documents: bool = False,
        max_chunks_per_doc: int | None = None,
        overlap_tokens: int | None = None,
    ) -> Dict[str, Any]:
        """Asynchronously rerank documents using SiliconFlow API.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            model: Model identifier for reranking.
            top_n: Number of top results to return. If None, returns all.
            instruction: Optional instruction to guide the reranking process.
            return_documents: Whether to return document texts in response.
            max_chunks_per_doc: Maximum chunks per document for long documents.
            overlap_tokens: Number of overlapping tokens between chunks.

        Returns:
            API response containing reranked results.

        Raises:
            httpx.HTTPStatusError: If API request fails.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "query": query,
            "documents": documents,
        }

        # Add optional parameters
        if top_n is not None:
            payload["top_n"] = top_n
        if instruction is not None:
            payload["instruction"] = instruction
        if return_documents:
            payload["return_documents"] = return_documents
        if max_chunks_per_doc is not None:
            payload["max_chunks_per_doc"] = max_chunks_per_doc
        if overlap_tokens is not None:
            payload["overlap_tokens"] = overlap_tokens

        response = await self.async_client.post("/rerank", json=payload)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        """Close synchronous client connection."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    async def aclose(self) -> None:
        """Close asynchronous client connection."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def __enter__(self) -> SiliconFlowClient:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    async def __aenter__(self) -> SiliconFlowClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.aclose()
