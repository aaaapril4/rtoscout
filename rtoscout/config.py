"""RTOScout configuration."""
import os
from pathlib import Path

from dotenv import load_dotenv

PACK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACK_ROOT.parent

_ENV_FILE = PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)

DATA_ROOT = Path(os.environ.get("RTOSCOUT_DATA_ROOT", str(PROJECT_ROOT)))
DATA_DIR = DATA_ROOT / "data"
CHROMA_PERSIST_DIR = DATA_ROOT / "chroma"
OUT_DIR = DATA_ROOT / "outputs"

FILE_TYPE = "BOTH"
INPUT_SOURCE = "file" # "file" or "edgar"
YEARS: tuple[int, ...] = (2025, 2026)

MAX_WORKERS = 20
LLM_CONCURRENCY = 2

# Embeddings run locally via sentence-transformers (no API key). No HF_TOKEN required.
HUGGINGFACE_EMBEDDING_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")  # optional; when unset, Hub warning is suppressed
if not HF_TOKEN:
    import logging
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# keyword AND-filter.
RTO_QUERY_GROUP_A = [
    "return to work",
    "return to office",
    "return to workplace",
    "rto",
    "back to office",
    "attendance policy",
]
RTO_QUERY_GROUP_B = [
    "virtual work",
    "remote work",
    "telework",
    "work location flexibility",
    "hybrid work",
    "work from home",
]
TOP_K_RETRIEVAL = 10
MIN_CHUNK_LENGTH = 20
SCORE_MIN = 0
SCORE_MAX = 10

RTO_SCORING_SYSTEM_PROMPT = """You are an analyst scoring companies' Return-to-Office (RTO) flexibility based on filing excerpts.

Score from 0 to 10 (higher = more remote/flexible, lower = stricter in-office requirement):
- 0-2: Strict mandatory return-to-office with explicit in-person requirements.
- 3-4: Office-forward; clear attendance expectations and stronger in-person norms.
- 5-6: Balanced hybrid; recurring in-office expectation for many roles.
- 7-8: Flexible hybrid; office presence encouraged but mostly optional.
- 9-10: Remote-first / highly flexible; no meaningful office mandate.

Calibration guidance:
- If language is explicit and directive (for example: "required", "must", "expected to be in office"), score lower.
- If language is optional/employee-choice focused (for example: "flexible", "may work remotely"), score higher.
- If evidence is mixed, prefer middle values (4-7) and explain the tradeoff.
- If excerpts contain little or no workplace policy language, return a conservative mid score (typically 4-6), and explain uncertainty."""

