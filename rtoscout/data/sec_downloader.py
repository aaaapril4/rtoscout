"""Download 10-K filings via sec-edgar-downloader."""
from pathlib import Path
from typing import Optional

from sec_edgar_downloader import Downloader

from ..schemas.models import CompanyInput


class SecDownloader:
    """SEC EDGAR 10-K downloader."""

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

    def download_10k(
        self,
        company: CompanyInput,
        limit: int = 1,
    ) -> Path:
        """
        Download 10-K for the given company (source=edgar; use ticker or cik).
        If company.year is set, download the 10-K for that fiscal year (after/before Jan 1–Dec 31).
        Returns the path to the downloaded 10-K directory (accession subdir).
        """
        self._get_downloader()
        cik = company.cik
        ticker = (company.ticker or company.company_id).strip().upper()
        if cik is None:
            cik = self._ticker_to_cik(ticker)
        if company.year is not None:
            after = f"{company.year}-01-01"
            before = f"{company.year}-12-31"
            self._downloader.get(
                "10-K", cik, limit=limit,
                after=after, before=before,
                download_details=True,
            )
        else:
            self._downloader.get("10-K", cik, limit=limit, download_details=True)
        for base in (
            self.download_dir / "sec-edgar-filings" / str(cik).zfill(10) / "10-K",
            Path(getattr(self._downloader, "save_dir", self.download_dir)) / str(cik).zfill(10) / "10-K",
            self.download_dir / str(cik).zfill(10) / "10-K",
        ):
            if base.exists():
                break
        else:
            base = self.download_dir / "sec-edgar-filings" / str(cik).zfill(10) / "10-K"

        subdirs = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
        if company.year is not None and subdirs:
            # Accession folder names are like 0001543151-25-000008; middle two digits encode the year (e.g. 25 → 2025).
            yy = str(company.year)[-2:]
            year_subdirs = [
                d for d in subdirs
                if "-" in d.name and d.name.split("-")[1].startswith(yy)
            ]
            if year_subdirs:
                return year_subdirs[0].resolve()

        return subdirs[0].resolve() if subdirs else base.resolve()

    def _ticker_to_cik(self, ticker: str) -> str:
        import requests
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": "RTOScout research@example.com"},
            timeout=30,
        )
        resp.raise_for_status()
        for entry in resp.json().values():
            if str(entry.get("ticker", "")).upper() == ticker:
                return str(entry["cik_str"])
        raise ValueError(f"Ticker not found: {ticker}")
