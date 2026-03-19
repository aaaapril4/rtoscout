"""Pydantic data models."""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class CompanyInput(BaseModel):
    """Single company input for download or local file."""

    company_id: str = Field(..., description="Unique id, e.g. ticker or CIK")
    ticker: str = Field(..., description="Ticker")
    source: Literal["file", "edgar"] = "edgar"
    path: Optional[str] = Field(None, description="Local 10-K path when source=file")
    cik: Optional[str] = None
    year: Optional[int] = Field(None, description="10-K year; default latest")


class DocumentChunk(BaseModel):
    """Text chunk with metadata."""

    content: str
    company_id: str
    company_name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RTOScoreResult(BaseModel):
    """RTO score result."""

    score: int = Field(ge=0, le=10, description="RTO strictness 0-10")
    rationale: str = Field(..., description="Score rationale")


class CompanyRTOOutput(BaseModel):
    """RTO analysis output for one company."""

    company_id: str
    company_name: str
    rto_score: int = Field(ge=0, le=10)
    rationale: str
