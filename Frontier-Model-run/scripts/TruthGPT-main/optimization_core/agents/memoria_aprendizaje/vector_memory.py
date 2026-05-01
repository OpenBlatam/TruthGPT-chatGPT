"""
OpenClaw Vector Memory — Pydantic-First Architecture.

High-level vector memory management using a Provider pattern with
typed configuration and structured search results.
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class VectorMemoryConfig(BaseModel):
    """Configuration for the Vector Memory subsystem."""
    db_path: str = Field(default="./openclaw_chroma_db", description="Path to the vector database")
    provider_type: str = Field(default="chroma", description="Backend provider: 'chroma' or 'noop'")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="SentenceTransformer model name")
    episodic_collection: str = "openclaw_episodic"
    semantic_collection: str = "openclaw_semantic"
    compaction_threshold: int = Field(default=15, description="Min episodes before compaction triggers")


class SearchResult(BaseModel):
    """Structured result from a vector memory search."""
    query: str
    collection: str
    documents: List[str] = Field(default_factory=list)
    count: int = 0


class MemoryAddResult(BaseModel):
    """Result of adding a document to vector memory."""
    memory_id: str
    collection: str
    success: bool


# ---------------------------------------------------------------------------
# Provider Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class VectorProvider(Protocol):
    """Interface for long-term vector memory providers."""
    async def add_document(self, collection: str, content: str, metadata: Dict[str, Any], doc_id: str) -> bool: ...
    async def query_documents(self, collection: str, query: str, where: Dict[str, Any], top_k: int) -> List[str]: ...
    async def delete_documents(self, collection: str, ids: List[str]) -> bool: ...
    async def get_all_documents(self, collection: str, where: Dict[str, Any]) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

class ChromaDBProvider:
    """Implementation of VectorProvider using ChromaDB."""

    def __init__(self, config: VectorMemoryConfig) -> None:
        import chromadb
        from chromadb.config import Settings
        self.client = chromadb.PersistentClient(
            path=config.db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collections: Dict[str, Any] = {}

        try:
            from chromadb.utils import embedding_functions
            self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config.embedding_model,
            )
            logger.info("ChromaDB: SentenceTransformers loaded (%s).", config.embedding_model)
        except ImportError:
            self.embed_fn = None
            logger.warning("ChromaDB: sentence-transformers missing.")

    def _get_collection(self, name: str) -> Any:
        if name not in self.collections:
            self.collections[name] = self.client.get_or_create_collection(
                name=name, embedding_function=self.embed_fn,
            )
        return self.collections[name]

    async def add_document(self, collection: str, content: str, metadata: Dict[str, Any], doc_id: str) -> bool:
        coll = self._get_collection(collection)
        await asyncio.to_thread(coll.add, documents=[content], metadatas=[metadata], ids=[doc_id])
        return True

    async def query_documents(self, collection: str, query: str, where: Dict[str, Any], top_k: int) -> List[str]:
        coll = self._get_collection(collection)
        results = await asyncio.to_thread(coll.query, query_texts=[query], n_results=top_k, where=where)
        docs = results.get("documents")
        return docs[0] if docs and docs[0] else []

    async def delete_documents(self, collection: str, ids: List[str]) -> bool:
        coll = self._get_collection(collection)
        await asyncio.to_thread(coll.delete, ids=ids)
        return True

    async def get_all_documents(self, collection: str, where: Dict[str, Any]) -> Dict[str, Any]:
        coll = self._get_collection(collection)
        return await asyncio.to_thread(coll.get, where=where)


class NoOpVectorProvider:
    """Fallback provider when no vector database is available."""
    async def add_document(self, *args: Any, **kwargs: Any) -> bool: return False
    async def query_documents(self, *args: Any, **kwargs: Any) -> List[str]: return []
    async def delete_documents(self, *args: Any, **kwargs: Any) -> bool: return False
    async def get_all_documents(self, *args: Any, **kwargs: Any) -> Dict[str, Any]: return {"ids": [], "documents": []}


# ---------------------------------------------------------------------------
# VectorMemory
# ---------------------------------------------------------------------------

class VectorMemory:
    """
    High-level Vector Memory management using a Provider pattern.

    Configuration is driven by ``VectorMemoryConfig``.
    """

    def __init__(
        self,
        db_path: str = "./openclaw_chroma_db",
        provider_type: str = "chroma",
        config: Optional[VectorMemoryConfig] = None,
    ) -> None:
        self.config = config or VectorMemoryConfig(db_path=db_path, provider_type=provider_type)
        self.provider: VectorProvider
        self.enabled = False

        if self.config.provider_type == "chroma":
            try:
                self.provider = ChromaDBProvider(self.config)
                self.enabled = True
            except Exception as e:
                logger.error("VectorMemory: ChromaDB init failed: %s", e)
                self.provider = NoOpVectorProvider()
        else:
            self.provider = NoOpVectorProvider()

    async def add_episodic(
        self,
        user_id: str,
        agent_name: str,
        event: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryAddResult:
        """Add an episodic memory entry."""
        mem_id = f"ep_{uuid.uuid4().hex[:12]}"
        if not self.enabled:
            return MemoryAddResult(memory_id=mem_id, collection=self.config.episodic_collection, success=False)

        meta = {"user_id": user_id, "agent": agent_name, "type": "episodic"}
        if metadata:
            meta.update(metadata)
        success = await self.provider.add_document(self.config.episodic_collection, event, meta, mem_id)
        return MemoryAddResult(memory_id=mem_id, collection=self.config.episodic_collection, success=success)

    async def add_semantic(
        self,
        user_id: str,
        agent_name: str,
        fact: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryAddResult:
        """Add a semantic memory entry."""
        mem_id = f"sm_{uuid.uuid4().hex[:12]}"
        if not self.enabled:
            return MemoryAddResult(memory_id=mem_id, collection=self.config.semantic_collection, success=False)

        meta = {"user_id": user_id, "agent": agent_name, "type": "semantic"}
        if metadata:
            meta.update(metadata)
        success = await self.provider.add_document(self.config.semantic_collection, fact, meta, mem_id)
        return MemoryAddResult(memory_id=mem_id, collection=self.config.semantic_collection, success=success)

    async def search_episodic(self, query: str, user_id: str, top_k: int = 3) -> SearchResult:
        """Search episodic memory and return a structured result."""
        if not self.enabled:
            return SearchResult(query=query, collection=self.config.episodic_collection)
        docs = await self.provider.query_documents(self.config.episodic_collection, query, {"user_id": user_id}, top_k)
        return SearchResult(query=query, collection=self.config.episodic_collection, documents=docs, count=len(docs))

    async def search_semantic(self, query: str, user_id: str, top_k: int = 3) -> SearchResult:
        """Search semantic memory and return a structured result."""
        if not self.enabled:
            return SearchResult(query=query, collection=self.config.semantic_collection)
        docs = await self.provider.query_documents(self.config.semantic_collection, query, {"user_id": user_id}, top_k)
        return SearchResult(query=query, collection=self.config.semantic_collection, documents=docs, count=len(docs))

    async def get_context_for_prompt(self, query: str, user_id: str) -> str:
        """Build a context string from relevant memories for prompt injection."""
        if not self.enabled:
            return ""
        episodes_result = await self.search_episodic(query, user_id, top_k=2)
        facts_result = await self.search_semantic(query, user_id, top_k=2)

        if not episodes_result.documents and not facts_result.documents:
            return ""

        context_parts = ["\n[Long-Term Memory Search Results]"]
        if facts_result.documents:
            context_parts.append("Relevant Semantic Facts / Rules:")
            for i, f in enumerate(facts_result.documents, 1):
                context_parts.append(f" {i}. {f}")
        if episodes_result.documents:
            context_parts.append("\nRelevant Past Episodes:")
            for i, e in enumerate(episodes_result.documents, 1):
                context_parts.append(f" {i}. {e}")
        return "\n".join(context_parts) + "\n"

    async def compact_episodic_memory(
        self, user_id: str, summarize_callback: Any, threshold: Optional[int] = None
    ) -> bool:
        """Compact old episodic memories into semantic summaries."""
        threshold = threshold or self.config.compaction_threshold
        if not self.enabled:
            return False
        try:
            results = await self.provider.get_all_documents(self.config.episodic_collection, {"user_id": user_id})
            ids, documents = results.get("ids", []), results.get("documents", [])
            if not ids or len(ids) < threshold:
                return False
            logger.info("VectorMemory: Compacting %d episodes for %s...", len(ids), user_id)
            full_text = "\n".join([f"- {doc}" for doc in documents])
            prompt = f"Resume estos eventos episódicos en hechos semánticos concisos:\n{full_text}"
            summary = await summarize_callback(prompt)
            await self.add_semantic(
                user_id=user_id, agent_name="MemoryCompactor",
                fact=summary, metadata={"compacted_from": len(ids)},
            )
            await self.provider.delete_documents(self.config.episodic_collection, ids)
            return True
        except Exception as e:
            logger.error("VectorMemory: Compaction failed: %s", e)
            return False

