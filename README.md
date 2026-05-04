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
poetry run python run.py -i data/companies.json
```

**`-i` / `--input`** — JSON with rows containing at least **`ticker`**; optional **`cik`**.

**Outputs:** **`OUT_DIR`** (default `<DATA_ROOT>/outputs`, `DATA_ROOT` defaults to project root) → **`results.csv`**, **`chunks.csv`**.

## Docker

From the repo root (where `.env` lives):

```bash
docker compose build
docker compose run --rm rtoscout python run.py -i /app/data/companies.json
```

Mount `./data` for input; optional **`./.env` → `/app/.env`** if you want `config` to read a file in-container. Chroma volume **`chroma`** → `/app/chroma`. For Ollama on the host, set `OLLAMA_BASE_URL=http://host.docker.internal:11434` in `.env` (Linux: often `http://172.17.0.1:11434`).

**Jupyter:** `docker compose --profile jupyter run --rm -p 8888:8888 jupyter` — token **`rtoscout`**; `notebooks/` is mounted.

## Repo layout

`rtoscout/` — config, data, index, engine, schemas, `pipeline.py`. Demo: `notebooks/rto_pipeline_demo.ipynb`.
