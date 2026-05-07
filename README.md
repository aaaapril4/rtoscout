# RTOScout 🚀

Sniffing out return-to-office (RTO) signals from 10-K and 10-Q filings using **RAG** and local **LLMs** (HuggingFace embeddings, ChromaDB, Ollama). ✨
> Huang, J. L., Huang, R., & Jie, Y. (submitted). Return to office reconsidered: HRM systems, public discourse, and the reconfiguration of work. Human Resource Management.

## 🛠 Setup

1. **Install Dependencies:**
   ```bash
   poetry install
   ```
2. **Pull the Model:** Install [Ollama](https://ollama.ai) and pull your LLM friend:
   ```bash
   ollama pull llama3.2
   ```
3. **Configure Env:** Create a `.env` in the project root with your technical vibes:
   * `OLLAMA_MODEL`, `OLLAMA_BASE_URL`
   * `RTOSCOUT_DATA_ROOT` (optional)
   * `HUGGINGFACE_EMBEDDING_MODEL`, `HF_TOKEN` (optional)

## 📊 The RTO Vibe Check

Signals extracted from 10-K and 10-Q filings to see which companies favor remote work versus a physical desk presence. ☕️ *Note: This is just for fun and based on vibe-parsing logic—please don't take it too seriously!*

| Tier | Samples 🌟 |
| :--- | :--- |
| **🌈 Tier 1** | COST, AAPL, ABNB, CRM, GOOGL, META, NFLX, NVDA, TSLA, UBER, WMT, ZM, A, AAMI, AAOI, AASP, ACAD, ACCS, ACIC, ACIW, ACLX, ACMR, ACN, ACT |
| **🍃 Tier 2** | BMY, CMG, COF, LMT, LYFT, MU, PEP, USB, AB, ABCL, ACEL, ACET, ACHC, ADAM, ADNT, ADPT, AENT, AEO, AGYS, AKBA |
| **☁️ Tier 3** | ADBE, DOW, KHC, RCL, TAP, STX, ACRS, ALMS, AMTB, AP, APLD, APYX, ARQ, AVR, BIRD, BLBD, BTSG, CBUS |
| **🐝 Tier 4** | AMZN, DASH, MCD, MSFT, ORCL, SBUX, UPS, V, ABEO, ACFN, ACHR, ACI, ACTG, ACVA, ADAC, ADP, ADVB, AEBI, AEMD, AEYE |
| **👔 Tier 5** | GS, DE, GM, JNJ, TGT, AACB, AARD, ABAT, ABCP, ABM, ABSI, ACBM, ACLS, ACNT, ACOG, ACR, ACXP |

## 🏃‍♀️ Run

Execute the pipeline with your favorite tickers! 📂💎

```bash
poetry run python run.py -i data/company.json
```

* **Inputs:** JSON containing at least **`ticker`** (and optional **`cik_str`**, **`title`**).
* **Outputs:** Check **`OUT_DIR`** (defaults to `<DATA_ROOT>/outputs`) for **`results.csv`** and **`chunks.csv`**.

## 🐳 Docker

Ready to containerize? Build and ship from the repo root:

```bash
docker compose build
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.2   # match OLLAMA_MODEL in .env
docker compose --profile cli run --rm rtoscout python run.py -i /app/company.json
```

* **Network:** If port **11434** is busy, stop host-level Ollama or adjust **`docker-compose.yml`**.
* **Jupyter Fun:** Access notebooks via `docker compose --profile jupyter run --rm -p 8888:8888 jupyter` (Token: **`rtoscout`**). 📓🍭