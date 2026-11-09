# Ask My Portfolio — RAG Q&A Chatbot

A Retrieval-Augmented Generation chatbot that answers questions about my resume,
project READMEs, and certificates — grounded in the actual source documents,
not the model's general knowledge.

**Live demo:** https://virajs-ask-my-portfolio.streamlit.app/
**Repo:** https://github.com/VirajBarapatre/ask-my-portfolio-rag

Built to close a real, recurring skill gap (RAG, vector databases, prompt
engineering, NLP) that kept showing up as a hard requirement in Data
Scientist / AI-ML job postings — and built so I can explain every design
decision in it, not just that it works.

---

## Status — all phases complete

- [x] Phase 0: Setup
- [x] Phase 1: Document ingestion & chunking
- [x] Phase 2: Embedding & vector store
- [x] Phase 3: Retrieval + generation pipeline
- [x] Phase 4: Streamlit UI
- [x] Phase 5: Free deployment (Streamlit Community Cloud + Groq)
- [x] Phase 6: NLP preprocessing (spaCy enrichment)

---

## Architecture

```
                    ┌─────────────────────────┐
                    │   data/                 │
                    │   resume/  readmes/     │
                    │   certificates/         │
                    └───────────┬─────────────┘
                                │
                     src/ingest.py
                     load + chunk (RecursiveCharacterTextSplitter)
                                │
                     src/nlp_enrich.py  (optional, Phase 6)
                     spaCy NER + skill PhraseMatcher → chunk metadata
                                │
                     src/embed_store.py
                     all-MiniLM-L6-v2 embeddings → FAISS index (persisted)
                                │
                                ▼
                    ┌─────────────────────────┐
User question ─────▶│   src/generate.py       │─────▶ grounded answer
                    │   retrieve → build      │       + cited sources
                    │   prompt → call_llm     │
                    └───────────┬─────────────┘
                                │
                    LLM_PROVIDER auto-switch:
                    - local dev  → Ollama (phi4-mini)
                    - deployed   → Groq (llama-3.1-8b-instant, free API)
                                │
                                ▼
                    src/app.py — Streamlit UI
                    "case file" design, stamp-style citations,
                    light/dark toggle, live index stats
```

---

## Project Structure

```
ask-my-portfolio-rag/
├── data/
│   ├── resume/            # resume PDF
│   ├── readmes/            # project README .md files
│   └── certificates/       # certificate summaries as .txt
├── faiss_index/             # persisted vector store (committed for deployment)
├── src/
│   ├── config.py            # all tunable settings + provider auto-detection
│   ├── ingest.py             # Phase 1: load + chunk documents
│   ├── nlp_enrich.py          # Phase 6: spaCy NER + skill tagging (metadata only)
│   ├── embed_store.py          # Phase 2: embeddings + FAISS build/query
│   ├── generate.py              # Phase 3: grounded prompt + LLM call (Ollama/Groq)
│   └── app.py                    # Phase 4: Streamlit UI
├── runtime.txt                    # pins Python 3.11 for Streamlit Cloud
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangChain | standard, well-documented chunking/loader utilities |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | free, local, 384-dim, no API cost |
| Vector store | FAISS | free, local, persists to disk |
| NLP enrichment | spaCy (NER + `PhraseMatcher`) | metadata only — see design notes below |
| Generation (local dev) | Ollama (`phi4-mini`) | free, fully offline |
| Generation (deployed) | Groq (`llama-3.1-8b-instant`) | free hosted API — Ollama can't run on Streamlit Cloud |
| UI | Streamlit | fast to build, free hosting via Community Cloud |

---

## Setup (local development)

```bash
# 1. Create and activate a virtual environment (Python 3.11 or 3.12 recommended —
#    avoid brand-new Python releases; ML packages lag behind on wheel support)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Ollama for local generation: https://ollama.com/download
ollama pull phi4-mini
ollama serve   # if not already running as a background service
```

## Add your own documents

- Resume PDF → `data/resume/`
- Project READMEs → `data/readmes/*.md`
- Certificate summaries → `data/certificates/*.txt`

## Run the pipeline

```bash
python -m src.ingest          # Phase 1 — sanity-check chunking
python -m src.nlp_enrich       # Phase 6 — see spaCy entities/skills on real chunks
python -m src.embed_store       # Phase 2 — build the FAISS index (also runs enrichment)
python -m src.generate           # Phase 3 — 5 test Q&A pairs end-to-end via terminal
streamlit run src/app.py          # Phase 4 — the actual UI, http://localhost:8501
```

If you edit anything in `data/`, re-run `python -m src.embed_store` to rebuild
the index before the UI reflects the change.

---

## Deployment (free — Streamlit Community Cloud + Groq)

Ollama needs a persistent local server with several GB of RAM, which free
hosting tiers don't support. The deployed version therefore uses **Groq's
free API** instead, auto-selected at runtime — no manual toggling.

1. Get a free Groq API key: https://console.groq.com (no card required)
2. Push the repo to GitHub, including the pre-built `faiss_index/` (committed
   deliberately, so the deployed app doesn't need to re-embed everything on
   a memory-constrained free tier)
3. On https://share.streamlit.io: **New app** → point at the repo, branch
   `main`, main file `src/app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your-key-here"
   ```
5. Deploy. `LLM_PROVIDER` in `src/config.py` detects `GROQ_API_KEY` in the
   environment and automatically switches from Ollama to Groq — same
   codebase, no branching logic needed elsewhere.

`runtime.txt` pins Python 3.11 — newer versions (3.13+) don't yet have
prebuilt wheels for `torch`/`numpy` on most platforms, which causes
pip to compile from source and exceed free-tier memory limits during install.

---

## Design decisions (interview notes)

**Chunk size (600 chars, 100 overlap) — tuned from a real failure, not
guessed upfront.** Initial testing at 400/50 caused a specific, reproducible
bug: a query for "what ML models has Viraj worked with" failed to retrieve
the resume chunk that explicitly lists `Isolation Forest, Random Forest`,
because the skills section is dense, paragraph-free text that got split
mid-category at 400 chars. Widening to 600/100 fixed it. This is a genuine
example of debugging RAG retrieval, not a default I copied.

**NLP preprocessing does NOT mutate the embedded text.** `src/nlp_enrich.py`
deliberately does not lowercase, strip stopwords, or stem chunk text —
that's a classic-IR technique that actively hurts modern sentence-transformer
embeddings, which are trained on natural, cased, punctuated text. Instead,
spaCy adds `entities` and `skills` as chunk *metadata*, sitting alongside the
untouched text — useful for future faceted filtering, with zero effect on
retrieval quality.

**Hallucination control is prompt-enforced and tested, not assumed.** The
system prompt in `generate.py` explicitly instructs the model to say "I
don't know based on the available documents" when context doesn't contain
an answer, and `TEST_QUESTIONS` includes one deliberately unanswerable
question ("Has Viraj worked with Kubernetes?") specifically to verify that
behavior actually holds — rather than trusting the prompt wording blindly.

**Two real generation failures found and fixed during testing:**
- A small local model (`phi4-mini`) inferred "Bash/PowerShell" from a Git
  mention despite an explicit "no unstated assumptions" instruction — fixed
  by adding a concrete negative example to the prompt rule.
- The same model occasionally mangled ALL-CAPS resume names during title-
  case conversion (e.g. "RAVINDRA" → "R Avindra") — fixed with an explicit
  proper-noun preservation rule.

Both are documented, reproducible examples of iterating on LLM output
quality, not just "the prompt works."

**Provider abstraction (`call_llm` in `generate.py`) exists specifically to
support free deployment.** Local dev uses Ollama; the deployed app can't, so
it needs a hosted API — but a paid one would violate the project's zero-cost
constraint. Groq's free tier resolves that without code branching: the same
`call_llm()` function routes based on whether `GROQ_API_KEY` is present in
the environment.

---

## What This Project Lets Me Honestly Claim

- RAG pipeline design and implementation (LangChain, FAISS)
- Embeddings and semantic search, including real retrieval tuning based on
  observed failures
- Prompt engineering: grounding, hallucination control, tested edge cases
- NLP preprocessing (spaCy NER + PhraseMatcher) applied as metadata
  enrichment, with a defensible reason for *not* doing destructive
  preprocessing
- End-to-end deployment: GitHub → Streamlit Community Cloud, with a working
  local/hosted provider-switching architecture

## What I Should Not Claim

- "Production" RAG experience — this is a personal demo project
- Model fine-tuning or training — everything here uses pre-trained models
  via API or local inference
- Advanced retrieval techniques (re-ranking, hybrid search, multi-hop) —
  not implemented; a known, articulable "what I'd improve next"

---

## Known limitations / future improvements

- **Retrieval is pure semantic search** — no hybrid (keyword + semantic)
  search or re-ranking. A query like "what ML models" occasionally misses a
  chunk containing the literal answer as a short parenthetical. Hybrid
  search (e.g. combining FAISS with BM25) is the natural next step.
- **spaCy NER accuracy is limited** on PDF-extracted text without clean
  sentence boundaries (e.g. `achieving35%` runs together). Entities are
  therefore metadata-only and never relied on for retrieval or generation
  correctness — only the exact-match `skills` field is.
- **Free-tier constraints**: Groq's free API and Streamlit Community Cloud
  both have rate/resource limits appropriate for a demo, not production
  traffic.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` when running `streamlit run` | Streamlit runs the file directly, not as a package | Already handled — `app.py` inserts the project root into `sys.path` at the top |
| `WinError 10061` / connection refused calling Ollama | Ollama server isn't running | `ollama serve` in a separate terminal |
| `pip install` fails compiling `numpy`/`torch` from source | Python version too new, no prebuilt wheels yet | Use Python 3.11 or 3.12 |
| `StreamlitSetPageConfigMustBeFirstCommandError` | Some import (e.g. touching `st.secrets`) ran before `set_page_config()` | Avoid calling any Streamlit command at module import time |
| Deployed app builds but crashes / OOM during `pip install` | Streamlit Cloud free tier ran out of memory compiling from source | Confirm `runtime.txt` pins Python 3.11 so wheels install instead of compiling |

---

## Author

Viraj Barapatre — Data Analyst / Business Analyst / Data Scientiest / ML Engineer
[GitHub](https://github.com/VirajBarapatre) ·
[LinkedIn](https://linkedin.com/in/viraj-barapatre) ·
[Portfolio](https://viraj-portfolio-three.vercel.app)
