# RTOScout

**RTOScout** — RTO (Return to Office) analysis from 10-K filings: **RAG** + **LLM** scoring. Local stack: HuggingFace embeddings, ChromaDB, Ollama. **No OpenAI (or other) API key needed** for embeddings; they run locally.

## Setup

- **Local:** Set up the project with **Poetry** (you manage `poetry install` etc.).
- **Run in container:** Use **Docker** (see below).

Requires [Ollama](https://ollama.ai) and a model (e.g. `ollama pull llama3.2`). Optional: copy `.env.example` to `.env` for `OLLAMA_MODEL` / `OLLAMA_BASE_URL`.

## Docker

```bash
docker compose build
docker compose run --rm -v $(pwd):/out rtoscout python main.py --tickers AAPL,MSFT --output /out/results.json
# Or: --input data/companies.json
```

### Docker Jupyter

From the project root:

```bash
# Build (once)
docker compose build

# Start Jupyter
docker compose --profile jupyter run --rm -p 8888:8888 jupyter
```

In the terminal you’ll see a URL like `http://127.0.0.1:8888/?token=rtoscout`. Open that in your browser to use the notebooks. The `notebooks/` folder is mounted, so edits are saved on your machine. If you see `ModuleNotFoundError: No module named 'rtoscout'`, rebuild the image: `docker compose build --no-cache` then start Jupyter again.

For the **LLM scoring** cells to work, Ollama must be reachable from the container. If Ollama runs on your host, create a `.env` with:

```bash
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

(On Linux you may need `http://172.17.0.1:11434` or host network.) To pass env vars from a `.env` file, run with `--env-file .env` before `run`.

## Usage

| Command | Description |
|--------|-------------|
| `poetry run python main.py --tickers AAPL,MSFT --output out.json` | Fetch 10-Ks from EDGAR and score |
| `poetry run python main.py --input data/companies.json --output out.json` | Companies from JSON |
| `poetry run python main.py --tickers-file data/tickers_sample.txt --output out.json` | All tickers from a file (one per line) |
| `poetry run python main.py --tickers AAPL,MSFT --years 2022,2023,2024 --output out.json` | Fetch 10-K for each year (company_id becomes e.g. AAPL_2023) |
| `poetry run python main.py --input data/companies.json --skip-index --output out.json` | Re-score only (reuse index) |
| `poetry run python main.py --tickers AAPL --save-chunks data/chunks_and_context.json` | Save processed chunks and retrieved context to one JSON file |

**Analyze many companies:** use a companies JSON or a ticker list file. To get **data from different years**, use `--years 2022,2023,2024` (expands each company into one entry per year), or set `"year": 2023` on a company in your JSON to fetch that year’s 10-K. Sample files: `data/companies_sample.json` (10 companies), `data/tickers_sample.txt` (one ticker per line). Add your own tickers to the file and run with `--tickers-file` or build a custom `data/companies.json`.

**Companies JSON** (e.g. `data/companies.json`): list of `{ "company_id", "company_name", "source": "edgar"|"file", "ticker"|"path" }`.

## Output

Per company: `company_id`, `company_name`, `rto_score` (0–10), `rationale`.  
0 = no RTO / flexible; 5 = hybrid; 10 = strict mandatory RTO.

## Config

`rtoscout/config.py` or env: `HUGGINGFACE_EMBEDDING_MODEL`, `HF_TOKEN` (optional, for Hub download limits), `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `CHUNK_SIZE`, `RTO_QUERIES`, etc.

**If you see an error about API key or quota/limit:** the app does not use OpenAI. Rebuild the Docker image (`docker compose build`) and re-run the notebook with a fresh kernel so the container uses the current code (local HuggingFace embeddings only).

## Layout

`rtoscout/{data,index,engine,schemas,config,pipeline}` — download/preprocess, embed, vector store, retriever, analyzer (Ollama), facade.

## Notebook

`notebooks/rto_pipeline_demo.ipynb` — run with `poetry run jupyter notebook` or Docker Jupyter profile.
