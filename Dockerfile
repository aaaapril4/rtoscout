# RTOScout: local RAG (HuggingFace + Ollama)
FROM python:3.11-slim

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false

# Install dependencies (use poetry.lock if present for reproducible builds)
COPY pyproject.toml poetry.lock* README.md ./
COPY rtoscout ./rtoscout/
RUN poetry install --no-interaction

# App
COPY main.py ./
COPY data ./data/

ENV PYTHONPATH=/app

CMD ["python", "main.py", "--help"]
