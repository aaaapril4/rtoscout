"""Pydantic data models."""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from ..config import FILE_TYPE


class CompanyInput(BaseModel):
    """Single company input for download or local file."""

    ticker: str = Field(..., description="Ticker")
    source: Literal["file", "edgar"] = "edgar"
    path: Optional[str] = Field(None, description="Local 10-K/10-Q path when source=file")
    cik: Optional[str] = None
    file_id: Optional[str] = Field(None, description="Filing identifier")
    file_type: str = Field(default=FILE_TYPE, description="SEC filing form, e.g. 10-K or 10-Q")


class DocumentChunk(BaseModel):
    """Text chunk with metadata."""

    content: str
    file_id: str
    ticker: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RTOScoreResult(BaseModel):
    """RTO score result."""

    score: int = Field(ge=0, le=10, description="RTO strictness 0-10")
    rationale: str = Field(..., description="Score rationale")


class CompanyRTOOutput(BaseModel):
    """RTO analysis output for one company."""

    file_id: str
    ticker: str
    rto_score: int = Field(ge=0, le=10)
    rationale: str
