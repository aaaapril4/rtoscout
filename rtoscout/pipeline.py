"""Facade: data → index → retrieve → score pipeline."""
import gc, shutil, time
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Set, Tuple

from .config import CHROMA_PERSIST_DIR, DATA_DIR, TOP_K_RETRIEVAL
from .data.preprocessor import Preprocessor
from .data.sec_downloader import SecDownloader
from .engine.analyzer import Analyzer
from .engine.retriever import Retriever
from .index.vector_store import VectorStore
from .schemas.models import CompanyInput, CompanyRTOOutput, DocumentChunk

class RTOPipeline:
    def __init__(self):
        self.persist_dir = Path(CHROMA_PERSIST_DIR)
        self.download_dir = Path(DATA_DIR / "sec_filings")
        self._preprocessor = Preprocessor()
        self._downloader = SecDownloader(download_dir=self.download_dir)
        self._vector_store = None

    def prepare_context_only(
        self,
        company: CompanyInput,
        years: List[int] = None
    ) -> Tuple[Optional[str], List[dict], str]:
        """
        PHASE 1: download/read → clean & chunk → index → retrieve
        Returns (combined_context, chunks, ticker).
        """
        all_chunks: List[DocumentChunk] = []

        base: Optional[Path] = None
        if company.source == "edgar":
            base = self._downloader.download(
                company, limit=1, file_type=company.file_type, years=years
            )
        elif company.source == "file" and company.path:
            base = Path(company.path)
        
        if not base:
            return None, [], company.ticker

        year_filter = set(years) if years else None
        filings = self._filings_from_base(base, company, year_filter, company.file_type)

        if not filings:
            return None, [], company.ticker

        file_ids_to_retrieve: dict = {}
        for filing in filings:
            chunks, year, file_id_used = self._load_and_chunk(filing)
            if year_filter and year not in year_filter:
                continue
            all_chunks.extend(chunks)
            file_ids_to_retrieve[file_id_used] = filing.file_type

        if not all_chunks:
            return None, [], filings[0].ticker

        self._vector_store = VectorStore.build_from_chunks(
            chunks=all_chunks,
            persist_directory=self.persist_dir,
        )
        
        self._retriever = Retriever(vector_store=self._vector_store, top_k=TOP_K_RETRIEVAL)
        ticker = filings[0].ticker
        chunks = []
        contexts: List[str] = []

        for file_id, file_type in file_ids_to_retrieve.items():
            context, chunk_rows = self._retriever.get_rto_context(file_id, ticker, file_type)
            if context:
                contexts.append(context)
            if chunk_rows:
                chunks.extend(chunk_rows)

        combined_context = "\n\n---\n\n".join(contexts) if contexts else ""
        
        if self._vector_store:
            try:
                self._vector_store.close()
            except:
                pass
        
        return combined_context, chunks, ticker

    def score_context(self, ticker: str, context: str) -> CompanyRTOOutput:
        """
        PHASE 2: Send context to Ollama for scoring.
        """
        analyzer = Analyzer()
        score_result = analyzer.score(ticker, context)
        return CompanyRTOOutput(
            ticker=ticker,
            rto_score=score_result.score,
            rationale=score_result.rationale,
        )

    def _load_and_chunk(self, company: CompanyInput) -> Tuple[List[DocumentChunk], int, str]:
        filing_path = Path(company.path)
        file_id_used = company.file_id or filing_path.name
        chunks, year = self._preprocessor.load_and_chunk_dir(
            filing_path,
            file_id=file_id_used,
            ticker=company.ticker,
            file_type=company.file_type
        )
        return chunks, year, file_id_used

    def _year_from_accession_dir_name(self, name: str) -> Optional[int]:
        try:
            parts = name.split("-")
            if len(parts) >= 2:
                return int("20" + parts[1][:2])
        except: 
            return None
        return None

    def _filings_from_base_helper(
        self,
        base: Path,
        template: CompanyInput,
        year_filter: Optional[Set[int]],
        file_type: str,
    ) -> List[CompanyInput]:
        base_path = base / file_type
        if not base_path.is_dir():
            return []
        out: List[CompanyInput] = []
        for p in sorted(base_path.iterdir()):
            if not p.is_dir():
                continue
            yr = self._year_from_accession_dir_name(p.name)
            if yr and (year_filter is None or yr in year_filter):
                out.append(
                    CompanyInput(
                        ticker=template.ticker,
                        source=template.source,
                        path=str(p.resolve()),
                        cik=template.cik,
                        file_id=p.name,
                        file_type=file_type,
                    )
                )
        return out

    def _filings_from_base(
        self,
        base: Path,
        template: CompanyInput,
        year_filter: Optional[Set[int]],
        file_type: str,
    ) -> List[CompanyInput]:
        if file_type == "BOTH":
            return self._filings_from_base_helper(base, template, year_filter, "10-K") + \
                   self._filings_from_base_helper(base, template, year_filter, "10-Q")
        return self._filings_from_base_helper(base, template, year_filter, file_type)


def prepare_company_worker(company: CompanyInput, years: List[int]):
    pipeline = RTOPipeline()
    temp_persist_dir = pipeline.persist_dir / company.ticker
    pipeline.persist_dir = temp_persist_dir
    
    try:
        context, chunks, ticker = pipeline.prepare_context_only(company, years)
        return ticker, context, chunks
    except Exception as e:
        print(f"[!] Error preparing {company.ticker}: {e}")
        return company.ticker, None, []
    finally:
        if hasattr(pipeline, "_vector_store") and pipeline._vector_store:
            try:
                pipeline._vector_store.close()
            except Exception as e:
                print(f"[!] Close error for {company.ticker}: {e}")

        del pipeline
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        if temp_persist_dir.exists():
            time.sleep(0.5) 
            shutil.rmtree(temp_persist_dir, ignore_errors=True)

def run_pipeline(
    companies: List[CompanyInput],
    year_list: List[int], 
    max_workers: int = 20,
    llm_concurrency: int = 2
):
    all_final_results: List[CompanyRTOOutput] = []
    all_chunks: List[dict] = []
    ready_for_scoring: List[Tuple[str, str]] = []

    with tqdm(total=len(companies), desc="Extracting & Indexing", unit="co") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(prepare_company_worker, c, year_list): c 
                for c in companies
            }

            for future in concurrent.futures.as_completed(futures):
                c = futures[future]
                try:
                    ticker, context, chunks = future.result()
                    if context:
                        ready_for_scoring.append((ticker, context))
                    if chunks:
                        all_chunks.extend(chunks)
                    pbar.set_postfix({"last_finished": c})
                except Exception as e:
                    tqdm.write(f"\n[!] {c} failed: {e}")
                pbar.update(1)

    if ready_for_scoring:
        scoring_pipeline = RTOPipeline()
        with tqdm(total=len(ready_for_scoring), desc="LLM Scoring", unit="co") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=llm_concurrency) as executor:
                scoring_futures = {
                    executor.submit(scoring_pipeline.score_context, t, c): t 
                    for t, c in ready_for_scoring
                }
                for future in concurrent.futures.as_completed(scoring_futures):
                    c = scoring_futures[future]
                    try:
                        all_final_results.append(future.result())
                    except Exception as e:
                        tqdm.write(f"\n[!] {c} failed: {e}")
                    pbar.update(1)

    return all_final_results, all_chunks