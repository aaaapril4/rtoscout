from rtoscout import run_pipeline
from rtoscout.schemas.models import CompanyInput
from rtoscout.config import DATA_DIR

import json

with open("/mnt/home/jieyaqi/code/RTOScout/company.json", 'r', encoding='utf-8') as f:
        company_data = json.load(f)

companies = []
cik_set = set()
secPath = DATA_DIR / "sec_filings" / "sec-edgar-filings"
for item in company_data.values():
    cik = str(item['cik_str']).zfill(10)
    if cik not in cik_set:
        comp = CompanyInput(
            ticker=item['ticker'],
            cik=str(item['cik_str']).zfill(10),
            source="file"
        )
        cik_set.add(cik)
    
        comp.path = str(secPath / comp.cik / "10-K")
        companies.append(comp)

run_pipeline(companies, [2020, 2021, 2022, 2023, 2024, 2025, 2026], 20)
