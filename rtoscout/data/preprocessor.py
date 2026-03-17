"""Clean HTML, extract text, and chunk 10-K content."""
from pathlib import Path
from typing import Optional
import re

from bs4 import BeautifulSoup

from ..schemas.models import DocumentChunk


class Preprocessor:
    """10-K text cleaning and chunking."""

    def __init__(self) -> None:
        pass

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

    def _merge_hr_separated_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """
        Merge paragraphs based on semantic cues and <hr> proximity.
        
        Rules:
        1. If a paragraph starts with a lowercase letter, merge it (highest priority).
        2. If an <hr> is nearby (either immediately before or immediately after), 
        check if the previous paragraph lacks terminal punctuation.
        3. Terminal punctuation includes: . ! ? " ” ) ]
        """
        if not paragraphs:
            return []

        def _is_hr(p: str) -> bool:
            s = p.strip().lower().replace(" ", "")
            return s.startswith("<hr")
        
        def _is_pure_symbol(p: str) -> bool:
            s = p.strip().lower().replace(" ", "")
            return bool(re.search(r'^[^a-zA-Z]+$', s))

        merged: list[str] = []
        n = len(paragraphs)
        
        for i in range(n):
            p_strip = paragraphs[i].strip()
            
            if not p_strip or _is_hr(p_strip) or _is_pure_symbol(p_strip):
                continue

            if not merged:
                merged.append(p_strip)
            else:
                prev = merged[-1]
                
                first_letter_match = re.search(r'[a-zA-Z]', p_strip)
            
                if first_letter_match:
                    is_continuation = first_letter_match.group().islower()
                else:
                    is_continuation = False
                
                has_hr_nearby = False
                
                if i > 0 and _is_hr(paragraphs[i-1]):
                    has_hr_nearby = True
                elif i > 1 and _is_hr(paragraphs[i-2]):
                    has_hr_nearby = True
                elif i + 1 < n and _is_hr(paragraphs[i+1]):
                    has_hr_nearby = True
                elif i + 2 < n and _is_hr(paragraphs[i+2]):
                    has_hr_nearby = True

                terminal_punctuations = ('.', '!', '?', '"', '”', ')', ']')
                is_unfinished = not prev.endswith(terminal_punctuations)
                
                if is_continuation or (has_hr_nearby and is_unfinished):
                    merged[-1] = " ".join(f"{prev} {p_strip}".split())
                else:
                    # Treat as a standalone paragraph
                    merged.append(p_strip)

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
        
        TARGET_CONTAINERS = {"div", "span", "p", "td", "th"}
        paragraphs: list[str] = []
        for el in soup.find_all(list(TARGET_CONTAINERS) + ["hr"]):
            name = getattr(el, "name", "").lower()
            if name == "hr":
                paragraphs.append("<hr>")
                continue
            
            if el.find(list(TARGET_CONTAINERS)):
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
