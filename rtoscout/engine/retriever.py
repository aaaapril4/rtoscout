"""Hybrid search: semantic multi-query over RTO-related terms."""
import re
from typing import Dict, List, Optional, Set

from ..config import (
    RTO_QUERY_GROUP_A,
    RTO_QUERY_GROUP_B,
    TOP_K_RETRIEVAL,
    MIN_CHUNK_LENGTH,
)
from ..index.vector_store import VectorStore


class Retriever:
    """RTO retrieval: multi-query semantic search over RTO-related queries.

    We rely purely on semantic similarity for relevance. To ensure snippets are
    related to both groups,we do two semantic passes (group A queries and group 
    B queries) and only keep snippets that appear in both result sets.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = TOP_K_RETRIEVAL,
        group_a: Optional[List[str]] = None,
        group_b: Optional[List[str]] = None,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self._queries_a = group_a or RTO_QUERY_GROUP_A
        self._queries_b = group_b or RTO_QUERY_GROUP_B

    def _build_company_chunks(
        self,
        file_id: str,
        ticker: str,
        file_type: str,
        block_parts: List[str],
    ) -> List[Dict[str, object]]:
        """Build one row per text block for batch writing."""
        rows = self.vector_store.get_all_chunks_for_files([file_id])
        filing_meta = next(((r.get("metadata") or {}) for r in rows if (r.get("metadata") or {})), {})
        blocks: List[Dict[str, object]] = []
        for part in block_parts:
            content = re.sub(r"\s+", " ", (part or "")).strip()
            if not content:
                continue

            blocks.append(
                {
                    "file_type": file_type,
                    "file_id": file_id,
                    "ticker": ticker,
                    "year": filing_meta.get("year"),
                    "source_url": filing_meta.get("source_url"),
                    "content": content,
                }
            )
        return blocks

    def get_rto_context(
        self,
        file_id: str,
        ticker: str,
        file_type: str,
    ) -> tuple[str, List[Dict[str, object]]]:
        """
        Retrieve RTO-related context via two-pass semantic search.
        """
        filter_dict = {"file_id": file_id}
        hits_a_indices: Set[int] = set()
        for query in self._queries_a:
            docs = self.vector_store.similarity_search(query, k=self.top_k, filter=filter_dict)
            for doc in docs:
                content = (doc.page_content or "").strip()
                if len(content.split()) < MIN_CHUNK_LENGTH:
                    continue
                idx = doc.metadata.get("chunk_index")
                if idx is not None:
                    hits_a_indices.add(int(idx))

        if not hits_a_indices:
            return "", []

        coverage_zone: Set[int] = set()
        for idx in hits_a_indices:
            coverage_zone.update({idx - 1, idx, idx + 1})

        hit_indices = set()
        for query in self._queries_b:
            docs = self.vector_store.similarity_search(query, k=self.top_k, filter=filter_dict)
            for doc in docs:
                content = (doc.page_content or "").strip()
                if len(content.split()) < MIN_CHUNK_LENGTH:
                    continue
                idx = doc.metadata.get("chunk_index")
                if idx is not None and int(idx) in coverage_zone:
                    chunk_idx = int(idx)
                    hit_indices.add(chunk_idx)
                    if chunk_idx - 1 in hits_a_indices:
                        hit_indices.add(chunk_idx - 1)
                    if chunk_idx + 1 in hits_a_indices:
                        hit_indices.add(chunk_idx + 1) 
        
        if not hit_indices:
            return "", []

        sorted_indices = sorted(list(hit_indices))
        block_parts = []
        
        prev = -999
        for i in sorted_indices:
            p = self.vector_store.get_chunk_content(file_id, i).strip()
            
            if not p:
                continue

            if not block_parts or i != prev + 1:
                block_parts.append(p)
            else:
                block_parts[-1] += "\n\n" + p
            
            prev = i

        chunk_rows = self._build_company_chunks(file_id, ticker, file_type, block_parts)
        if not chunk_rows:
            return "", []

        context = "\n\n---\n\n".join(block_parts) if block_parts else ""
        return context, chunk_rows