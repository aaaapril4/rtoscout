"""Clean HTML, extract text, and chunk 10-K content."""
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

from ..schemas.models import DocumentChunk


class Preprocessor:
    """10-K text cleaning and chunking."""

    def __init__(self) -> None:
        pass

    def extract_text_from_file(self, path: str | Path) -> str:
        """Extract and clean text from a local file (HTML or plain text)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        raw = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix.lower() in (".html", ".htm"):
            return self._clean_html_to_string(raw)
        return self._normalize_whitespace(raw)

    def extract_text_from_dir(self, dir_path: str | Path) -> str:
        """Find the primary document in a 10-K download dir and extract text (usually .htm)."""
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        primary = self._find_primary_html(dir_path)
        return self.extract_text_from_file(primary)

    def _find_primary_html(self, dir_path: Path) -> Path:
        """Return path to primary .htm/.html in directory (exclude index)."""
        candidates = list(dir_path.glob("*.htm")) + list(dir_path.glob("*.html"))
        candidates = [c for c in candidates if "index" not in c.name.lower()]
        if not candidates:
            candidates = list(dir_path.glob("*.htm")) + list(dir_path.glob("*.html"))
        if not candidates:
            raise FileNotFoundError(f"No .htm/.html in {dir_path}")
        return candidates[0]

    def _clean_html_to_string(self, html: str) -> str:
        """Legacy: extract all text from HTML as one string (used when not chunking from div/span)."""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

    def _strip_style_attributes(self, soup: BeautifulSoup) -> None:
        """Remove all style attributes from every element in the tree."""
        for tag in soup.find_all(True):
            if tag.has_attr("style"):
                del tag["style"]

    def _merge_hr_separated_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """
        Merge paragraphs that are split only by an <hr> marker.
        This is mainly for plain-text inputs or HTML-derived lists that contain
        literal <hr> markers.
        """

        def _is_hr(p: str) -> bool:
            s = p.strip().lower().replace(" ", "")
            return s.startswith("<hr")

        if not paragraphs:
            return paragraphs

        merged: list[str] = []
        i = 0
        n = len(paragraphs)
        while i < n:
            p = paragraphs[i]
            if _is_hr(p):
                # If an <hr> is between two paragraphs, merge next into previous.
                if merged and i + 1 < n:
                    # Only merge when the previous paragraph is likely still ongoing.
                    # Heuristic: if it ends with a period, treat it as a full sentence/paragraph and do not merge.
                    prev = merged[-1].rstrip()
                    if not prev.endswith("."):
                        next_p = paragraphs[i + 1]
                        merged[-1] = (prev + " " + next_p.lstrip()).strip()
                        i += 2
                        continue
                # Otherwise just skip the <hr>
                i += 1
                continue
            merged.append(p)
            i += 1
        return merged

    def _extract_paragraphs_from_html(self, html: str) -> list[str]:
        """
        Extract one paragraph per <div> or <span> in document order.

        We also track <hr> between blocks as a soft separator by inserting a
        literal \"<hr>\" marker into the paragraph stream. That marker is then
        consumed by _merge_hr_separated_paragraphs to optionally join adjacent
        paragraphs that are visually split by an <hr> but logically belong
        together.
        """
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        self._strip_style_attributes(soup)
        paragraphs: list[str] = []
        for el in soup.find_all(["div", "span", "hr"]):
            name = getattr(el, "name", "").lower()
            if name == "hr":
                paragraphs.append("<hr>")
                continue
            text = el.get_text(separator=" ", strip=True)
            text = " ".join(text.split())
            if text:
                paragraphs.append(text)
        paragraphs = self._merge_hr_separated_paragraphs(paragraphs)
        paragraphs = [p for p in paragraphs if not p.strip().lower().replace(" ", "").startswith("<hr")]
        return paragraphs

    def _normalize_whitespace(self, text: str) -> str:
        """Strip lines and drop empty; keep one newline between lines."""
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())

    def chunk(
        self,
        text: str,
        company_id: str,
        company_name: Optional[str] = None,
        year: Optional[int] = None,
    ) -> list[DocumentChunk]:
        """Split plain text into chunks by paragraph (double newline)."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        paragraphs = self._merge_hr_separated_paragraphs(paragraphs)
        return [
            DocumentChunk(
                content=p,
                company_id=company_id,
                company_name=company_name,
                metadata={"chunk_index": i},
            )
            for i, p in enumerate(paragraphs)
        ]

    def load_and_chunk_file(
        self,
        path: str | Path,
        company_id: str,
        company_name: Optional[str] = None,
        year: Optional[int] = None,
    ) -> list[DocumentChunk]:
        """Read file and chunk: HTML by div/span, plain text by double newline."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        raw = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix.lower() in (".html", ".htm"):
            paragraphs = self._extract_paragraphs_from_html(raw)
            paragraphs = self._merge_hr_separated_paragraphs(paragraphs)
            return [
                DocumentChunk(
                    content=p,
                    company_id=company_id,
                    company_name=company_name,
                    metadata={"chunk_index": i},
                )
                for i, p in enumerate(paragraphs)
            ]
        text = self._normalize_whitespace(raw)
        return self.chunk(text, company_id, company_name, year=year)

    def load_and_chunk_dir(
        self,
        dir_path: str | Path,
        company_id: str,
        company_name: Optional[str] = None,
        year: Optional[int] = None,
    ) -> list[DocumentChunk]:
        """Read primary HTML in directory and chunk by div/span."""
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        primary = self._find_primary_html(dir_path)
        return self.load_and_chunk_file(primary, company_id, company_name, year=year)
