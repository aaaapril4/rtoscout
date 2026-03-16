"""Embedding model wrapper (HuggingFace sentence-transformers)."""
from typing import List, Protocol

from ..config import HF_TOKEN, HUGGINGFACE_EMBEDDING_MODEL


class Embedder(Protocol):
    """Embedding interface."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


class HuggingFaceEmbedder:
    """HuggingFace text embeddings (local sentence-transformers, no API key)."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or HUGGINGFACE_EMBEDDING_MODEL
        self._model = None

    def _get_model(self):
        if self._model is None:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*HF Hub.*|.*HF_TOKEN.*", category=UserWarning)
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, token=HF_TOKEN)
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get_model().encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._get_model().encode([text], convert_to_numpy=True)[0].tolist()


def get_default_embedder() -> Embedder:
    """Return the default embedder (HuggingFace)."""
    return HuggingFaceEmbedder()
