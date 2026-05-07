#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from rtoscout import run_pipeline
from rtoscout.config import (
    DATA_DIR,
    OUT_DIR,
    INPUT_SOURCE,
    LLM_CONCURRENCY,
    MAX_WORKERS,
    YEARS,
    FILE_TYPE,
)
from rtoscout.schemas.models import CompanyInput


def _file_type() -> str:
    ft = FILE_TYPE
    return ft if ft in ("10-K", "10-Q", "BOTH") else "10-K"

def _source() -> str:
    src = INPUT_SOURCE
    return src if src in ("file", "edgar") else "edgar"


def _path_from_cik(cik: str) -> Path:
    return DATA_DIR / "sec_filings" / "sec-edgar-filings" / cik

def _companies_from_json(json_path: Path) -> list[CompanyInput]:
    """SEC-style map: ``cik_str``, ``ticker`` per row. Source is ``INPUT_SOURCE`` in config."""
    
    with open(json_path, encoding='utf-8') as f:
        raw: Any = json.load(f)

    def one(key: Any, row: Any) -> CompanyInput:
        if "ticker" not in row:
            raise SystemExit(f'--input key {key!r}: object with "ticker" required')
        comp = CompanyInput(
            ticker=str(row["ticker"]).strip().upper(),
            source=_source(),
            file_type=_file_type(),
        )
        
        if "cik_str" in row:
            comp.cik = str(row["cik_str"]).strip().zfill(10)
        if comp.source == "file":
            if not comp.cik:
                raise SystemExit(
                    f'--input key {key!r}: "cik" required when INPUT_SOURCE is "file" (see rtoscout/config.py)'
                )
            comp.path = str(_path_from_cik(comp.cik))

        return comp

    return [one(k, row) for k, row in raw.items()]


def main() -> None:
    p = argparse.ArgumentParser(
        description="RTOScout — retrieve RTO-related SEC filing excerpts and score with a local LLM (Ollama)."
    )
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help=(
            'SEC-style JSON object: keys -> rows with "ticker"'
        ),
    )
    args = p.parse_args()

    companies = _companies_from_json(args.input)

    if not companies:
        raise SystemExit("No companies to process")

    results, chunks = run_pipeline(
        companies,
        list(YEARS),
        max_workers=MAX_WORKERS,
        llm_concurrency=LLM_CONCURRENCY,
    )

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if chunks:
        pd.DataFrame(chunks).to_csv(out_dir / "chunks.csv", index=False)
    if results:
        pd.DataFrame([r.model_dump() for r in results]).to_csv(out_dir / "results.csv", index=False)

if __name__ == "__main__":
    main()
