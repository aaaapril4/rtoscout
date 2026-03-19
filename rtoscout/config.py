"""RTOScout configuration."""
import os
from pathlib import Path

from dotenv import load_dotenv

PACK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACK_ROOT.parent
DATA_DIR = Path('/mnt/scratch/jieyaqi/rtoscout') / "data"
CHROMA_PERSIST_DIR = Path('/mnt/scratch/jieyaqi/rtoscout') / "chroma_rto"

# Load .env from project root, then cwd. In Docker use env_file so vars are in the process env.
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    load_dotenv(_env_file)
load_dotenv()  # fallback: .env in current working directory

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
TOP_K_RETRIEVAL = 8
MIN_CHUNK_LENGTH = 5
SCORE_MIN = 0
SCORE_MAX = 10
