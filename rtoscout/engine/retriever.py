"""Hybrid search: semantic multi-query over RTO-related terms."""
from typing import Dict, List, Optional, Set

from ..config import (
    DATA_DIR,
    RTO_QUERY_GROUP_A,
    RTO_QUERY_GROUP_B,
    TOP_K_RETRIEVAL,
    MIN_CHUNK_LENGTH,
    FILE_TYPE
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

    def _save_company_context(
        self,
        company_id: str,
        company_name: str,
        context: str,
    ) -> None:
        """Write one context file per company_id, including the SEC source URL."""
        if not context:
            return

        source_url = None
        rows = self.vector_store.get_all_chunks_for_companies([company_id])
        for row in rows:
            meta = row.get("metadata") or {}
            source_url = meta.get("source_url")
            year = meta.get("year")
            if source_url:
                break

        out_dir = DATA_DIR / "rto_outputs" / FILE_TYPE
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{company_name}_{year}.txt"

        with out_file.open("w", encoding="utf-8") as f:
            f.write(f"company_name: {company_name}\n")
            f.write(f"year: {year}\n")
            if source_url:
                f.write(f"source_url: {source_url}\n")
            f.write("\n==== Retrieved Context ====\n\n")
            f.write(context)

    def get_rto_context(
        self,
        company_id: str,
        company_name: str,
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
            return ""

        sorted_indices = sorted(list(hit_indices))
        block_parts = []
        
        prev = -999
        for i in sorted_indices:
            p = self.vector_store.get_chunk_content(company_id, i).strip()
            
            if not p:
                continue

            if not block_parts or i != prev + 1:
                block_parts.append(p)
            else:
                block_parts[-1] += "\n\n" + p
            
            prev = i
            
        context = "\n\n---\n\n".join(block_parts) if block_parts else ""
        self._save_company_context(company_id, company_name, context)
        return block_parts