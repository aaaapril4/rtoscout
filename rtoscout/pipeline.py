"""Facade: data → index → retrieve → score pipeline."""
import json, gc, shutil, time
import concurrent.futures
import torch
from pathlib import Path
from tqdm import tqdm
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
                chunks = self._load_and_chunk(company)
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
        print("start retrieving")
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
        name = company.ticker
        filing_path: Optional[Path] = None

        if company.source == "edgar":
            filing_path = Path(self._downloader.download_10k(company))
        elif company.source == "file" and company.path:
            filing_path = Path(company.path)

        if filing_path is not None:
            chunks = self._preprocessor.load_and_chunk_dir(filing_path, cid, name)
            return chunks
        raise ValueError(f"Company {cid}: no filing path found")

def process_single_company_worker(company: CompanyInput, years: List[int]):
    pipeline = RTOPipeline()
    temp_persist_dir = pipeline.persist_dir / company.ticker
    pipeline.persist_dir = temp_persist_dir
    
    try:
        pipeline.run(company, years)
        return

    except Exception as e:
        print(f"[!] Error processing {company.ticker}: {str(e)}")
        return

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
    company_json_path: str, 
    year_list: List[int], 
    max_workers: int = 3
):
    with open(company_json_path, 'r', encoding='utf-8') as f:
            company_data = json.load(f)

    companies = [
        CompanyInput(
            ticker=item['ticker'],
            company_id=item['ticker'],
            cik=str(item['cik_str'])
        ) for item in company_data.values()
    ]

    total_tasks = len(companies)
    all_final_results = []

    with tqdm(total=total_tasks, desc="Analyzing RTO", unit="company") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_company = {
                executor.submit(process_single_company_worker, comp, year_list): comp 
                for comp in companies
            }
            
            for future in concurrent.futures.as_completed(future_to_company):
                comp = future_to_company[future]
                try:
                    data = future.result()
                    if data:
                        all_final_results.extend(data)
                    pbar.set_postfix({"last_finished": comp})
                except Exception as e:
                    tqdm.write(f"\n[!] {comp} failed: {e}")
                pbar.update(1)

    return all_final_results