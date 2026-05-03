from rtoscout import run_pipeline
from rtoscout.schemas.models import CompanyInput
from rtoscout.config import DATA_DIR, FILE_TYPE
import pandas as pd

import json

if __name__ == "__main__":
    with open("/mnt/home/jieyaqi/code/RTOScout/company.json", 'r', encoding='utf-8') as f:
        company_data = json.load(f)

    companies = []
    cik_set = set()
    secPath = DATA_DIR / "sec_filings" / "sec-edgar-filings"
    for item in company_data.values():
        cik = str(item['cik_str']).zfill(10)
        ticker = item['ticker']
        if cik not in cik_set:
            comp = CompanyInput(
                ticker=item['ticker'],
                company_id=item['ticker'],
                cik=str(item['cik_str']).zfill(10),
                source="file",
                file_type=FILE_TYPE
            )
            cik_set.add(cik)
        
            comp.path = str(secPath / comp.cik)
            companies.append(comp)

    results, chunks = run_pipeline(companies, [2025, 2026], max_workers=20, llm_concurrency=2)
    out_dir = DATA_DIR / "rto_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    if chunks:
        pd.DataFrame(chunks).to_csv(out_dir / "chunks.csv", index=False)
    if results:
        pd.DataFrame([r.dict() for r in results]).to_csv(out_dir / "results.csv", index=False)