"""Vector store operations (ChromaDB; extensible to FAISS / Qdrant)."""
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.documents import Document

from ..config import CHROMA_PERSIST_DIR
from ..schemas.models import DocumentChunk
from .embedder import Embedder, get_default_embedder


def _embedder_adapter(embedder: Embedder):
    from langchain_core.embeddings import Embeddings
    class Adapter(Embeddings):
        def __init__(self, e: Embedder):
            self._e = e
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return self._e.embed_documents(texts)
        def embed_query(self, text: str) -> List[float]:
            return self._e.embed_query(text)
    return Adapter(embedder)


class VectorStore:
    """Vector store (ChromaDB)."""

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str = "rto_10k",
        embedder: Optional[Embedder] = None,
    ):
        self.persist_directory = Path(persist_directory or CHROMA_PERSIST_DIR)
        self.collection_name = collection_name
        self._embedder = embedder or get_default_embedder()
        self._store = None

    def _get_chroma(self):
        if self._store is not None:
            return self._store
        from langchain_chroma import Chroma
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._store = Chroma(
            collection_name=self.collection_name,
            embedding_function=_embedder_adapter(self._embedder),
            persist_directory=str(self.persist_directory),
        )
        return self._store

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector store."""
        docs = [
            Document(
                page_content=c.content,
                metadata={
                    "company_id": c.company_id,
                    "company_name": c.company_name or c.company_id,
                    **c.metadata,
                },
            )
            for c in chunks
        ]
        self._get_chroma().add_documents(docs)

    @classmethod
    def build_from_chunks(
        cls,
        chunks: List[DocumentChunk],
        persist_directory: str | Path | None = None,
        collection_name: str = "rto_10k",
        embedder: Optional[Embedder] = None,
    ) -> "VectorStore":
        """Build vector store from all chunks (overwrites existing collection). Returns VectorStore instance."""
        from langchain_chroma import Chroma
        persist_directory = Path(persist_directory or CHROMA_PERSIST_DIR)
        persist_directory.mkdir(parents=True, exist_ok=True)
        embedder = embedder or get_default_embedder()
        docs = [
            Document(
                page_content=c.content,
                metadata={
                    "company_id": c.company_id,
                    "company_name": c.company_name or c.company_id,
                    **c.metadata,
                },
            )
            for c in chunks
        ]
        Chroma.from_documents(
            documents=docs,
            embedding=_embedder_adapter(embedder),
            persist_directory=str(persist_directory),
            collection_name=collection_name,
        )
        return cls.load(persist_directory=persist_directory, collection_name=collection_name, embedder=embedder)

    def similarity_search(
        self,
        query: str,
        k: int = 8,
        filter: Optional[dict[str, Any]] = None,
    ) -> List[Document]:
        """Semantic similarity search."""
        return self._get_chroma().similarity_search(query, k=k, filter=filter or {})

    def get_chunk_content(self, company_id: str, chunk_index: int) -> Optional[str]:
        """Return page_content of the chunk with given company_id and chunk_index, or None if not found."""
        chroma = self._get_chroma()
        try:
            # langchain_chroma Chroma exposes _collection (chromadb Collection)
            coll = getattr(chroma, "_collection", None) or getattr(chroma, "collection", None)
            if coll is None:
                return None
            result = coll.get(
                where={"company_id": company_id, "chunk_index": chunk_index},
                limit=1,
            )
            if not result or not result.get("documents"):
                return None
            doc = result["documents"][0]
            return doc if isinstance(doc, str) else (doc[0] if doc else None)
        except Exception:
            return None

    def get_all_chunks_for_companies(self, company_ids: List[str]) -> List[dict[str, Any]]:
        """Return all stored chunks for the given company_ids as list of dicts (content, company_id, company_name, metadata)."""
        if not company_ids:
            return []
        chroma = self._get_chroma()
        try:
            coll = getattr(chroma, "_collection", None) or getattr(chroma, "collection", None)
            if coll is None:
                return []
            result = coll.get(
                where={"company_id": {"$in": list(company_ids)}},
                include=["documents", "metadatas"],
            )
            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []
            out: List[dict[str, Any]] = []
            for i, doc in enumerate(documents):
                meta = (metadatas[i] or {}) if i < len(metadatas) else {}
                content = doc[0] if isinstance(doc, list) else doc
                if isinstance(content, list):
                    content = content[0] if content else ""
                out.append({
                    "content": content or "",
                    "company_id": meta.get("company_id", ""),
                    "company_name": meta.get("company_name"),
                    "metadata": {k: v for k, v in meta.items() if k not in ("company_id", "company_name")},
                })
            return out
        except Exception:
            return []

    @classmethod
    def load(
        cls,
        persist_directory: str | Path | None = None,
        collection_name: str = "rto_10k",
        embedder: Optional[Embedder] = None,
    ) -> "VectorStore":
        """Load an existing vector store."""
        store = cls(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedder=embedder,
        )
        store._get_chroma()
        return store
