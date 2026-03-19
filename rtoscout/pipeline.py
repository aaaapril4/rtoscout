"""Facade: data → index → retrieve → score pipeline."""
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .config import CHROMA_PERSIST_DIR, DATA_DIR, TOP_K_RETRIEVAL
from .data.preprocessor import Preprocessor
from .data.sec_downloader import SecDownloader
# from .engine.analyzer import Analyzer
from .engine.retriever import Retriever
from .index.vector_store import VectorStore
from .schemas.models import CompanyInput, CompanyRTOOutput, DocumentChunk


class RTOPipeline:
    """RTO analysis pipeline facade for one company"""

    def __init__(self):
        self.persist_dir = Path(CHROMA_PERSIST_DIR)
        self.download_dir = Path(DATA_DIR / "sec_filings")
        self._preprocessor = Preprocessor()
        self._downloader = SecDownloader(download_dir=self.download_dir)

    def run(
        self,
        company: CompanyInput,
        years: List[int],
        skip_index: bool = False,
    ) -> List[CompanyRTOOutput]:
        """
        Full pipeline: download/read → clean & chunk → index → retrieve → LLM score for one company.
        skip_index=True uses existing vector store and only re-runs retrieve + score.
        """
        all_chunks: List[DocumentChunk] = []
        if years is not None:
            companies = [CompanyInput(company_id=f"{company.company_id}_{year}", ticker=company.ticker, source=company.source, year=year, path=company.path, cik=company.cik) for year in years]
        else:
            companies = [company]

        if not skip_index:
            for company in companies:
                chunks, cid = self._load_and_chunk(company)
                all_chunks.extend(chunks)
            if all_chunks:
                self._vector_store = VectorStore.build_from_chunks(
                    chunks=all_chunks,
                    persist_directory=self.persist_dir,
                )

        if self._vector_store is None:
            self._vector_store = VectorStore.load(persist_directory=self.persist_dir)

        self._retriever = Retriever(vector_store=self._vector_store, top_k=TOP_K_RETRIEVAL)
        # self._analyzer = Analyzer()

        # results: List[CompanyRTOOutput] = []
        for company in companies:
            context = self._retriever.get_rto_context(company.company_id, company.ticker)
            # score_result = self._analyzer.score_rto(name, cid, context)
            # results.append(CompanyRTOOutput(
            #     company_id=company.company_id,
            #     company_name=name,
            #     rto_score=score_result.score,
            #     rationale=score_result.rationale,
            # ))

        # return results
        return None
    

    def _load_and_chunk(self, company: CompanyInput) -> Tuple[List[DocumentChunk], str]:
        """Load 10-K and chunk according to company config. Returns (chunks, cid_used) so run() can use the same cid for retrieval."""
        cid = company.company_id
        name = company.company_name
        filing_path: Optional[Path] = None

        if company.source == "edgar":
            filing_path = Path(self._downloader.download_10k(company))
        elif company.source == "file" and company.path:
            filing_path = Path(company.path)

        if filing_path is not None:
            chunks = self._preprocessor.load_and_chunk_dir(filing_path, cid, name)
            return (chunks, cid)
        raise ValueError(f"Company {cid}: no filing path found")
