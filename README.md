# RTOScout

Return-to-office signals from 10-K filings using **RAG** + a local **LLM** (HuggingFace embeddings, ChromaDB, Ollama).

## Setup

```bash
poetry install
```

Install [Ollama](https://ollama.ai) and pull a model (e.g. `ollama pull llama3.2`).

**Env:** put **`.env` only in the project root** (next to `pyproject.toml`). Typical vars: `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, optional `RTOSCOUT_DATA_ROOT`, `HUGGINGFACE_EMBEDDING_MODEL`, `HF_TOKEN`.

## Run

```bash
poetry run python run.py -i data/company.json
```

**`-i` / `--input`** — JSON with rows containing at least **`ticker`**; optional **`cik`**.

**Outputs:** **`OUT_DIR`** (default `<DATA_ROOT>/outputs`, `DATA_ROOT` defaults to project root) → **`results.csv`**, **`chunks.csv`**.

## Docker

From the repo root (where `.env` lives):

```bash
docker compose build
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.2   # once; match OLLAMA_MODEL in .env
docker compose --profile cli run --rm rtoscout python run.py -i /app/company.json
```

Stop Ollama on the host if port **11434** is already in use, or change the left-hand side of **`ports`** for **`ollama`** in **`docker-compose.yml`**.

**Paths in the container:** **`./data`** → **`/app/data`**. **`company.json`** → **`/app/company.json`**. **`.env`** → **`/app/.env`**. Chroma: **`./chroma`** → **`/app/chroma`**.

**Jupyter**: `docker compose --profile jupyter run --rm -p 8888:8888 jupyter` — token **`rtoscout`**.

## Repo layout

`rtoscout/` — config, data, index, engine, schemas, `pipeline.py`. Demo: `notebooks/rto_pipeline_demo.ipynb`.
