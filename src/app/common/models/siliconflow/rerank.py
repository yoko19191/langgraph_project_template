"""SiliconFlow Rerank document compressor for LangChain."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Sequence, Union

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.utils import get_from_dict_or_env
from pydantic import ConfigDict, Field, model_validator

from .client import SiliconFlowClient


class SiliconflowRerank(BaseDocumentCompressor):
    """Document compressor using SiliconFlow Rerank API.

    This compressor uses SiliconFlow's rerank models to reorder documents
    based on their relevance to a query.

    Example:
        >>> from langchain_core.documents import Document
        >>> reranker = SiliconflowRerank(
        ...     model="BAAI/bge-reranker-v2-m3",
        ...     top_n=3
        ... )
        >>> docs = [
        ...     Document(page_content="Apple is a fruit"),
        ...     Document(page_content="Banana is yellow")
        ... ]
        >>> compressed = reranker.compress_documents(docs, "fruit")

    Attributes:
        model: Model identifier for reranking. Defaults to BAAI/bge-reranker-v2-m3.
        top_n: Number of documents to return after reranking.
        api_key: SiliconFlow API key. Reads from SILICONFLOW_API_KEY if not provided.
        base_url: Base URL for SiliconFlow API.
        instruction: Optional instruction to guide the reranking process.
        return_documents: Whether to return document texts in API response.
        max_chunks_per_doc: Maximum chunks per document for long documents.
        overlap_tokens: Number of overlapping tokens between chunks.
        client: SiliconFlow client instance (auto-initialized).
    """

    model: str = "BAAI/bge-reranker-v2-m3"
    """Model to use for reranking."""

    top_n: int = 3
    """Number of documents to return."""

    api_key: str | None = Field(None, alias="siliconflow_api_key")
    """SiliconFlow API key. Must be specified directly or via SILICONFLOW_API_KEY env var."""

    base_url: str = "https://api.siliconflow.cn/v1"
    """Base URL for SiliconFlow API."""

    instruction: str | None = None
    """Optional instruction to guide the reranking process."""

    return_documents: bool = False
    """Whether to return document texts in API response."""

    max_chunks_per_doc: int | None = None
    """Maximum chunks per document for long documents."""

    overlap_tokens: int | None = None
    """Number of overlapping tokens between chunks."""

    client: SiliconFlowClient | None = None
    """SiliconFlow client instance."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that API key exists and initialize client.

        Args:
            values: Field values from initialization.

        Returns:
            Updated values with initialized client.

        Raises:
            ValueError: If API key is not found.
        """
        if not values.get("client"):
            # Get API key from values or environment
            api_key = get_from_dict_or_env(
                values, "api_key", "SILICONFLOW_API_KEY", default=None
            )

            if not api_key:
                raise ValueError(
                    "SiliconFlow API key must be provided either through "
                    "api_key/siliconflow_api_key parameter or "
                    "SILICONFLOW_API_KEY environment variable"
                )

            # Initialize client
            values["client"] = SiliconFlowClient(
                api_key=api_key,
                base_url=values.get("base_url", "https://api.siliconflow.cn/v1"),
            )
            values["api_key"] = api_key

        return values

    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        top_n: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Rerank documents and return results with scores.

        Args:
            documents: Sequence of documents to rerank.
            query: Query string for reranking.
            top_n: Number of results to return. If None, uses self.top_n.

        Returns:
            List of dicts with 'index' and 'relevance_score' keys.

        Example:
            >>> results = reranker.rerank(
            ...     documents=["doc1", "doc2"],
            ...     query="search query",
            ...     top_n=1
            ... )
            >>> # [{"index": 0, "relevance_score": 0.95}]
        """
        if len(documents) == 0:
            return []

        # Extract text from documents
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]

        # Determine top_n value
        effective_top_n = top_n if top_n is not None else self.top_n

        # Call API
        response = self.client.rerank(
            query=query,
            documents=docs,
            model=self.model,
            top_n=effective_top_n,
            instruction=self.instruction,
            return_documents=self.return_documents,
            max_chunks_per_doc=self.max_chunks_per_doc,
            overlap_tokens=self.overlap_tokens,
        )

        # Parse results
        result_dicts = []
        for res in response.get("results", []):
            result_dicts.append(
                {
                    "index": res["index"],
                    "relevance_score": res.get(
                        "relevance_score", res.get("score", 0.0)
                    ),
                }
            )

        return result_dicts

    async def arerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        top_n: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Asynchronously rerank documents and return results with scores.

        Args:
            documents: Sequence of documents to rerank.
            query: Query string for reranking.
            top_n: Number of results to return. If None, uses self.top_n.

        Returns:
            List of dicts with 'index' and 'relevance_score' keys.
        """
        if len(documents) == 0:
            return []

        # Extract text from documents
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]

        # Determine top_n value
        effective_top_n = top_n if top_n is not None else self.top_n

        # Call API asynchronously
        response = await self.client.arerank(
            query=query,
            documents=docs,
            model=self.model,
            top_n=effective_top_n,
            instruction=self.instruction,
            return_documents=self.return_documents,
            max_chunks_per_doc=self.max_chunks_per_doc,
            overlap_tokens=self.overlap_tokens,
        )

        # Parse results
        result_dicts = []
        for res in response.get("results", []):
            result_dicts.append(
                {
                    "index": res["index"],
                    "relevance_score": res.get(
                        "relevance_score", res.get("score", 0.0)
                    ),
                }
            )

        return result_dicts

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Compress documents using SiliconFlow's rerank API.

        This is the main LangChain interface for document compression.

        Args:
            documents: Sequence of documents to compress.
            query: Query string for ranking relevance.
            callbacks: Optional callbacks (currently unused).

        Returns:
            Sequence of reranked documents with relevance scores in metadata.

        Example:
            >>> docs = [Document(page_content="text1"), Document(page_content="text2")]
            >>> compressed = reranker.compress_documents(docs, "query")
            >>> print(compressed[0].metadata["relevance_score"])
        """
        if len(documents) == 0:
            return []

        compressed = []
        for res in self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)

        return compressed

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Asynchronously compress documents using SiliconFlow's rerank API.

        This provides native async support instead of using run_in_executor.

        Args:
            documents: Sequence of documents to compress.
            query: Query string for ranking relevance.
            callbacks: Optional callbacks (currently unused).

        Returns:
            Sequence of reranked documents with relevance scores in metadata.
        """
        if len(documents) == 0:
            return []

        compressed = []
        for res in await self.arerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)

        return compressed
