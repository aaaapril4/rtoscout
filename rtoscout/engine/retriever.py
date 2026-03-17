"""Hybrid search: semantic multi-query over RTO-related terms."""
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

    def get_rto_context(
        self,
        company_id: str,
        company_name: Optional[str] = None,
    ) -> str:
        """
        Retrieve RTO-related context via two-pass semantic search.
        """

        filter_dict = {"company_id": company_id}
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
            return ""

        coverage_zone: Set[int] = set()
        for idx in hits_a_indices:
            coverage_zone.update({idx - 1, idx, idx + 1})

        final_hits: Dict[int, str] = {}
        for query in self._queries_b:
            docs = self.vector_store.similarity_search(query, k=self.top_k, filter=filter_dict)
            for doc in docs:
                content = (doc.page_content or "").strip()
                if len(content.split()) < MIN_CHUNK_LENGTH:
                    continue
                idx = doc.metadata.get("chunk_index")
                if idx is None or int(idx) not in coverage_zone:
                    continue
                
                chunk_idx = int(idx)
                if chunk_idx in final_hits:
                    continue

                prev = self.vector_store.get_chunk_content(company_id, chunk_idx - 1) if chunk_idx - 1 in hits_a_indices else ""
                next_ = self.vector_store.get_chunk_content(company_id, chunk_idx + 1) if chunk_idx + 1 in hits_a_indices else ""
                
                block_parts = [p for p in (prev, content, next_) if p and p.strip()]
                final_hits[chunk_idx] = "\n\n".join(p.strip() for p in block_parts)

        seen_content: Set[str] = set()
        parts: List[str] = []
        
        for idx in sorted(final_hits.keys()):
            block = final_hits[idx]
            if block in seen_content:
                continue
            seen_content.add(block)
            parts.append(block)

        return "\n\n---\n\n".join(parts) if parts else ""