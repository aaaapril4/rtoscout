"""Facade: data → index → retrieve → score pipeline."""
import gc, shutil, time
import concurrent.futures
import torch
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
    """RTO analysis pipeline facade for one company"""

    def __init__(self):
        self.persist_dir = Path(CHROMA_PERSIST_DIR)
        self.download_dir = Path(DATA_DIR / "sec_filings")
        self._preprocessor = Preprocessor()
        self._downloader = SecDownloader(download_dir=self.download_dir)
        self._vector_store = None

    def run(
        self,
        company: CompanyInput,
        years: List[int] = None
    ) -> Tuple[List[CompanyRTOOutput], List[dict]]:
        """
        Full pipeline: download/read → clean & chunk → index → retrieve → LLM score for one company.
        """
        all_chunks: List[DocumentChunk] = []

        base: Optional[Path] = None
        if company.source == "edgar":
            base = self._downloader.download(
                company, limit=1, file_type=company.file_type, years=years
            )
        elif company.source == "file" and company.path:
            base = Path(company.path)
        else:
            raise ValueError("Provide company.path (filing base dir) for source=file, or use source=edgar.")

        year_filter: Optional[Set[int]]
        if years is not None:
            year_filter = set(years)
        else:
            year_filter = None

        filings = self._filings_from_base(base, company, year_filter, company.file_type)

        if not filings:
            return [], []

        file_ids_to_retrieve: dict = {}
        for filing in filings:
            chunks, year, file_id_used = self._load_and_chunk(filing)
            if year_filter and year not in year_filter:
                continue
            all_chunks.extend(chunks)
            file_ids_to_retrieve[file_id_used] = filing.file_type

        if not file_ids_to_retrieve:
            return [], []

        if all_chunks:
            self._vector_store = VectorStore.build_from_chunks(
                chunks=all_chunks,
                persist_directory=self.persist_dir,
            )

        if self._vector_store is None:
            self._vector_store = VectorStore.load(persist_directory=self.persist_dir)

        self._retriever = Retriever(vector_store=self._vector_store, top_k=TOP_K_RETRIEVAL)
        self._analyzer = Analyzer()
        results: List[CompanyRTOOutput] = []
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
        score_result = self._analyzer.score_rto(ticker, combined_context)
        results.append(CompanyRTOOutput(
            ticker=ticker,
            rto_score=score_result.score,
            rationale=score_result.rationale,
        ))

        print("finish processing ", ticker)
        return results, chunks
    

    def _load_and_chunk(self, company: CompanyInput) -> Tuple[List[DocumentChunk], int, str]:
        """Load one accession filing dir and chunk it. Returns (chunks, year, file_id_used)."""
        if not company.path:
            raise ValueError("Filing input must have path set to an accession directory.")
        filing_path = Path(company.path)
        file_id_used = company.file_id or filing_path.name

        chunks, year = self._preprocessor.load_and_chunk_dir(
            filing_path,
            file_id=file_id_used,
            ticker=company.ticker,
            file_type=company.file_type,
        )
        return chunks, year, file_id_used

    def _year_from_accession_dir_name(self, name: str) -> Optional[int]:
        """Map accession folder name like 0001543151-25-000008 to calendar year (e.g. 2025)."""
        try:
            parts = name.split("-")
            if len(parts) < 2:
                return None
            yy = parts[1]
            if len(yy) >= 2 and yy[:2].isdigit():
                return int("20" + yy[:2])
        except (ValueError, IndexError):
            return None
        return None


    def _filings_from_base_helper(
        self,
        base: Path,
        template: CompanyInput,
        year_filter: Optional[Set[int]],
        file_type: str,
    ) -> List[CompanyInput]:
        """List accession subdirs under a filing-type base (…/10-K or …/10-Q)."""
        base = base / file_type
        if not base.is_dir():
            raise NotADirectoryError(f"Filing base is not a directory: {base}")

        out: List[CompanyInput] = []
        for p in sorted(base.iterdir()):
            if not p.is_dir():
                continue
            yr = self._year_from_accession_dir_name(p.name)
            if yr is None:
                continue
            if year_filter is not None and yr not in year_filter:
                continue
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
            return self._filings_from_base_helper(base, template, year_filter, "10-K") + self._filings_from_base_helper(base, template, year_filter, "10-Q")
        else:
            return self._filings_from_base_helper(base, template, year_filter, file_type)

def process_single_company_worker(company: CompanyInput, years: List[int]):
    pipeline = RTOPipeline()
    temp_persist_dir = pipeline.persist_dir / company.ticker
    pipeline.persist_dir = temp_persist_dir
    
    try:
        return pipeline.run(company, years)

    except Exception as e:
        print(f"[!] Error processing {company.ticker}: {str(e)}")
        return [], []

    finally:
        if hasattr(pipeline, "_vector_store") and pipeline._vector_store:
            try:
                pipeline._vector_store.close()
            except Exception as e:
                print(f"[!] Close error for {company.ticker}: {e}")

        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if temp_persist_dir.exists():
            time.sleep(0.5) 
            shutil.rmtree(temp_persist_dir, ignore_errors=True)


def run_pipeline(
    companies: List[CompanyInput],
    year_list: List[int], 
    max_workers: int = 3
):

    total_tasks = len(companies)
    all_final_results: List[CompanyRTOOutput] = []
    all_chunks: List[dict] = []

    with tqdm(total=total_tasks, desc="Analyzing RTO", unit="company") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_company = {
                executor.submit(process_single_company_worker, comp, year_list): comp 
                for comp in companies
            }
            
            for future in concurrent.futures.as_completed(future_to_company):
                comp = future_to_company[future]
                try:
                    results, chunk_rows = future.result()
                    if results:
                        all_final_results.extend(results)
                    if chunk_rows:
                        all_chunks.extend(chunk_rows)
                    pbar.set_postfix({"last_finished": comp})
                except Exception as e:
                    tqdm.write(f"\n[!] {comp} failed: {e}")
                pbar.update(1)

    return all_final_results, all_chunks