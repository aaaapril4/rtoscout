"""Hybrid search: semantic multi-query over RTO-related terms."""
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from ..config import (
    RTO_QUERY_GROUP_A,
    RTO_QUERY_GROUP_B,
    TOP_K_RETRIEVAL,
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

        1) Run semantic similarity for all group A queries (\"return to office\"
           style) and collect hits per (company_id, chunk_index).
        2) Run semantic similarity for all group B queries (remote / hybrid /
           WFH style) and collect hits per (company_id, chunk_index).
        3) Intersect the two key sets; keep only snippets that are retrieved by
           at least one query from each group. Each hit is expanded to include
           the previous and next chunks (when available) for context.
        """
        # First pass: semantic hits for group A queries (broad RTO candidates)
        # Second pass: semantic hits for group B queries, but only within the
        # candidate set discovered by the first pass.
        hits_a: Dict[Tuple[str, int], str] = {}
        hits_b: Dict[Tuple[str, int], str] = {}
        filter_dict = {"company_id": company_id}

        def _collect_hits(
            queries: List[str],
            allowed_keys: Optional[set[Tuple[str, int]]] = None,
        ) -> Dict[Tuple[str, int], str]:
            out: Dict[Tuple[str, int], str] = {}
            for query in queries:
                docs = self.vector_store.similarity_search(
                    query,
                    k=self.top_k,
                    filter=filter_dict,
                )
                for doc in docs:
                    content = (doc.page_content or "").strip()
                    if not content:
                        continue
                    chunk_index = doc.metadata.get("chunk_index")
                    if chunk_index is None:
                        # No index; we still keep it but key on (-1) to avoid collisions.
                        key = (company_id, -1)
                    else:
                        key = (company_id, int(chunk_index))
                    if allowed_keys is not None and key not in allowed_keys:
                        # Skip anything outside the candidate set from the first pass.
                        continue
                    if key in out:
                        continue
                    # Build the block (current ±1) for richer context when index is known.
                    if chunk_index is not None:
                        prev = self.vector_store.get_chunk_content(company_id, int(chunk_index) - 1)
                        next_ = self.vector_store.get_chunk_content(company_id, int(chunk_index) + 1)
                        block_parts = [p for p in (prev, content, next_) if p and (p or "").strip()]
                        block = "\n\n".join(p.strip() for p in block_parts if p.strip())
                    else:
                        block = content
                    if not block:
                        continue
                    out[key] = block
            return out

        # Pass 1: broad RTO candidates from group A
        hits_a = _collect_hits(self._queries_a)
        if not hits_a:
            return ""

        # Pass 2: refine using group B, but only among the group A candidates
        hits_b = _collect_hits(self._queries_b, allowed_keys=set(hits_a.keys()))
        if not hits_b:
            return ""

        seen_content: set[str] = set()
        parts: List[str] = []
        for key, block in hits_b.items():
            if not block:
                continue
            if block in seen_content:
                continue
            seen_content.add(block)
            parts.append(block)

        return "\n\n---\n\n".join(parts) if parts else ""

    # Keyword-based filters remain removed; we enforce \"must relate to both\"
    # concepts via two independent semantic passes and intersection.
