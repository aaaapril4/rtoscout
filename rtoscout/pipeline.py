"""Facade: data → index → retrieve → score pipeline."""
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .config import CHROMA_PERSIST_DIR
from .data.preprocessor import Preprocessor
from .data.sec_downloader import SecDownloader
from .engine.analyzer import Analyzer
from .engine.retriever import Retriever
from .index.vector_store import VectorStore
from .schemas.models import CompanyInput, CompanyRTOOutput, DocumentChunk


class RTOPipeline:
    """RTO analysis pipeline facade."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        download_dir: Optional[Path] = None,
    ):
        self.persist_dir = Path(persist_dir or CHROMA_PERSIST_DIR)
        self.download_dir = Path(download_dir or "data/sec_filings")
        self._preprocessor = Preprocessor()
        self._downloader = SecDownloader(download_dir=self.download_dir)
        self._vector_store: Optional[VectorStore] = None
        self._retriever: Optional[Retriever] = None
        self._analyzer: Optional[Analyzer] = None

    def run(
        self,
        companies: List[CompanyInput],
        skip_index: bool = False,
        save_chunks_and_context_path: Optional[Path] = None,
    ) -> List[CompanyRTOOutput]:
        """
        Full pipeline: download/read → clean & chunk → index → retrieve → LLM score.
        skip_index=True uses existing vector store and only re-runs retrieve + score.
        If save_chunks_and_context_path is set, write one JSON with processed_chunks and retrieved_context_by_company.
        """
        all_chunks: List[DocumentChunk] = []
        # (company, cid_used_for_chunks) so retrieval uses the same cid as indexing
        company_and_cid: List[Tuple[CompanyInput, str]] = []

        if not skip_index:
            for company in companies:
                chunks, cid = self._load_and_chunk(company)
                all_chunks.extend(chunks)
                company_and_cid.append((company, cid))
            if all_chunks:
                self._vector_store = VectorStore.build_from_chunks(
                    chunks=all_chunks,
                    persist_directory=self.persist_dir,
                )
        else:
            for company in companies:
                cid = company.company_id
                if company.year is not None:
                    suffix = f"_{company.year}"
                    if not cid.endswith(suffix):
                        cid = f"{cid}_{company.year}"
                company_and_cid.append((company, cid))

        if self._vector_store is None:
            self._vector_store = VectorStore.load(persist_directory=self.persist_dir)

        self._retriever = Retriever(vector_store=self._vector_store)
        self._analyzer = Analyzer()

        results: List[CompanyRTOOutput] = []
        retrieved_context_by_company: dict[str, str] = {}
        for company, cid in company_and_cid:
            name = company.company_name or company.company_id
            context = self._retriever.get_rto_context(cid, name)
            retrieved_context_by_company[cid] = context
            score_result = self._analyzer.score_rto(name, cid, context)
            results.append(CompanyRTOOutput(
                company_id=company.company_id,
                company_name=name,
                rto_score=score_result.score,
                rationale=score_result.rationale,
            ))

        if save_chunks_and_context_path:
            self._write_chunks_and_context(
                companies=companies,
                all_chunks=all_chunks,
                retrieved_context_by_company=retrieved_context_by_company,
                path=Path(save_chunks_and_context_path),
            )
        return results

    def _write_chunks_and_context(
        self,
        companies: List[CompanyInput],
        all_chunks: List[DocumentChunk],
        retrieved_context_by_company: dict[str, str],
        path: Path,
    ) -> None:
        """Write processed chunks and retrieved context to a single JSON file.
        Processed chunks are always included when saving, even if a company has no
        retrieved matches (retrieved_context_by_company[company_id] may be "").
        """
        if all_chunks:
            processed_chunks = [c.model_dump() for c in all_chunks]
        else:
            company_ids = [c.company_id for c in companies]
            stored = self._vector_store.get_all_chunks_for_companies(company_ids)
            processed_chunks = stored
        payload = {
            "processed_chunks": processed_chunks,
            "retrieved_context_by_company": retrieved_context_by_company,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_and_chunk(self, company: CompanyInput) -> Tuple[List[DocumentChunk], str]:
        """Load 10-K and chunk according to company config. Returns (chunks, cid_used) so run() can use the same cid for retrieval."""
        base_cid = company.company_id
        name = company.company_name or base_cid
        year = company.year
        filing_path: Optional[Path] = None

        if company.source == "edgar":
            filing_path = Path(self._downloader.download_10k(company))
        elif company.source == "file" and company.path:
            filing_path = Path(company.path)

        if filing_path is not None and filing_path.exists():
            primary_path = (
                self._preprocessor._find_primary_html(filing_path)
                if filing_path.is_dir()
                else filing_path
            )
            doc_year = self._preprocessor.extract_fiscal_year_from_document(primary_path)
            if doc_year is not None:
                year = doc_year
                # Strip trailing _YYYY from base_cid so we can attach document year
                parts = base_cid.split("_")
                base = parts[0] if (len(parts) >= 2 and parts[-1].isdigit()) else base_cid
                cid = f"{base}_{year}"
            else:
                cid = base_cid
                if year is not None:
                    suffix = f"_{year}"
                    if not cid.endswith(suffix):
                        cid = f"{cid}_{year}"
        else:
            cid = base_cid
            if year is not None:
                suffix = f"_{year}"
                if not cid.endswith(suffix):
                    cid = f"{cid}_{year}"

        if company.source == "edgar" and filing_path is not None:
            chunks = self._preprocessor.load_and_chunk_dir(filing_path, cid, name, year=year)
            return (chunks, cid)
        if company.source == "file" and company.path:
            path = Path(company.path)
            if path.is_dir():
                chunks = self._preprocessor.load_and_chunk_dir(path, cid, name, year=year)
            else:
                chunks = self._preprocessor.load_and_chunk_file(path, cid, name, year=year)
            return (chunks, cid)
        raise ValueError(f"Company {base_cid}: need source=edgar or source=file with path")


def run_pipeline(
    companies: List[dict[str, Any]],
    persist_dir: Optional[str | Path] = None,
    skip_index: bool = False,
    output_path: Optional[str | Path] = None,
    save_chunks_and_context_path: Optional[str | Path] = None,
) -> List[dict[str, Any]]:
    """
    Convenience entry: companies is a list of dicts compatible with CompanyInput.
    If output_path is set, write results to JSON or CSV.
    If save_chunks_and_context_path is set, write one JSON with processed_chunks and retrieved_context_by_company.
    """
    inputs = [CompanyInput(**c) for c in companies]
    pipeline = RTOPipeline(persist_dir=persist_dir)
    save_path = Path(save_chunks_and_context_path) if save_chunks_and_context_path else None
    outputs = pipeline.run(inputs, skip_index=skip_index, save_chunks_and_context_path=save_path)
    results = [
        {"company_id": o.company_id, "company_name": o.company_name, "rto_score": o.rto_score, "rationale": o.rationale}
        for o in outputs
    ]
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".json":
            import json
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            import csv
            with output_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["company_id", "company_name", "rto_score", "rationale"])
                w.writeheader()
                w.writerows(results)
    return results
