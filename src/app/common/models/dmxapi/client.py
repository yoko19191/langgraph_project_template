"""DMXAPI Client for Rerank."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import httpx


class DMXAPIClient:
    """DMXAPI client supporting both sync and async operations.

    Provides methods to call DMXAPI's rerank API with supported parameters.
    Note: DMXAPI supports basic rerank parameters only (model, query, documents, top_n).

    Args:
        api_key: DMXAPI API key. If not provided, reads from DMXAPI_API_KEY env var.
        base_url: Base URL for DMXAPI. Defaults to official endpoint.
        timeout: Request timeout in seconds.

    Example:
        >>> client = DMXAPIClient()
        >>> result = client.rerank(
        ...     query="search query",
        ...     documents=["doc1", "doc2"],
        ...     model="qwen3-reranker-8b"
        ... )
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://www.dmxapi.cn/v1",
        timeout: float = 30.0,
    ):
        """Initialize DMXAPI client."""
        self.api_key = api_key or os.getenv("DMXAPI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DMXAPI API key must be provided either through "
                "api_key parameter or DMXAPI_API_KEY environment variable"
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
        model: str = "qwen3-reranker-8b",
        top_n: int | None = None,
    ) -> Dict[str, Any]:
        """Rerank documents synchronously using DMXAPI.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            model: Model identifier for reranking.
            top_n: Number of top results to return. If None, returns all.

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

        if top_n is not None:
            payload["top_n"] = top_n

        response = self.sync_client.post("/rerank", json=payload)
        response.raise_for_status()
        return response.json()

    async def arerank(
        self,
        query: str,
        documents: List[str],
        model: str = "qwen3-reranker-8b",
        top_n: int | None = None,
    ) -> Dict[str, Any]:
        """Rerank documents asynchronously using DMXAPI.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            model: Model identifier for reranking.
            top_n: Number of top results to return. If None, returns all.

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

        if top_n is not None:
            payload["top_n"] = top_n

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

    def __enter__(self) -> DMXAPIClient:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    async def __aenter__(self) -> DMXAPIClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.aclose()
