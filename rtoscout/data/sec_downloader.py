"""Download 10-K/10-Q filings via sec-edgar-downloader."""
from pathlib import Path
from typing import Optional, List

from sec_edgar_downloader import Downloader

from ..schemas.models import CompanyInput


class SecDownloader:
    """SEC EDGAR 10-K/10-Q downloader."""

    def __init__(
        self,
        download_dir: Optional[Path] = None,
    ):
        self.download_dir = Path(download_dir) if download_dir else Path("data/sec_filings")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._downloader = None

    def _get_downloader(
        self,
        company_name: str = "RTOScout",
        email: str = "jieyaqi@msu.edu",
    ):
        if self._downloader is None:
            self._downloader = Downloader(
            company_name=company_name,
            email_address=email,
            download_folder=str(self.download_dir),
        )


    def _resolve_cik(self, company: CompanyInput) -> str:
        cik = company.cik
        ticker = company.ticker.strip().upper()
        if cik is None:
            cik = self._ticker_to_cik(ticker)
        return str(cik).zfill(10)
    
    def _download_helper(
        self,
        years: Optional[List[int]],
        file_type: str,
        cik: str,
        limit: int = 1,
    ) -> None:
        if years is not None:
            after = f"{min(years)}-01-01"
            before = f"{max(years)}-12-31"
            self._downloader.get(
                file_type, cik,
                after=after, before=before,
                download_details=True,
            )
        else:
            self._downloader.get(file_type, cik, limit=limit, download_details=True)
        return

    def download(
        self,
        company: CompanyInput,
        limit: int = 1,
        file_type: str = "10-K",
        years: Optional[List[int]] = None,
    ) -> Path:
        """
        Download one batch of 10-K/10-Q for the given company (ticker or cik).
        """
        self._get_downloader()
        cik = self._resolve_cik(company)
        if file_type == "BOTH":
            for ft in ["10-K", "10-Q"]:
                self._download_helper(years, ft, cik, limit)
        else:
            self._download_helper(years, file_type, cik, limit)
        return self.download_dir / "sec-edgar-filings" / cik
    
    def _ticker_to_cik(self, ticker: str) -> str:
        import requests
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": "RTOScout jieyaqi@msu.edu"},
            timeout=30,
        )
        resp.raise_for_status()
        for entry in resp.json().values():
            if str(entry.get("ticker", "")).upper() == ticker:
                return str(entry["cik_str"])
        raise ValueError(f"Ticker not found: {ticker}")
