# GitLab Handbook Assistant

A conversational RAG (Retrieval-Augmented Generation) chatbot that lets GitLab employees and candidates ask questions about the [GitLab Handbook](https://handbook.gitlab.com) and [Direction pages](https://about.gitlab.com/direction/), with conversation memory enabling follow-up questions.

Built with **Streamlit · LangChain · Groq (Llama 3.3 70B) · FAISS · HuggingFace Embeddings**.

## Table of Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Tech stack & why](#tech-stack--why)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [How the index works](#how-the-index-works)
- [How conversation memory works](#how-conversation-memory-works)
- [Adding more pages](#adding-more-pages)
- [Deployment](#deployment-streamlit-community-cloud)
- [Guardrails & transparency](#guardrails--transparency)
- [Troubleshooting](#troubleshooting)

---

## What it does

GitLab's Handbook is one of the most comprehensive public company operating manuals in the world — thousands of pages covering values, engineering culture, leadership, people operations, and product direction. Finding specific information requires either expert knowledge of the structure or lengthy manual searching.

This chatbot makes the handbook **conversational**:

- Ask questions in plain English: _"What is GitLab's approach to remote work?"_
- Ask follow-ups: _"How does that apply to managers?"_ , the bot understands the context from the previous message
- Every answer cites the exact handbook page it retrieved from, improving **transparency**
- The bot says "I don't know" rather than hallucinating when information isn't in the index

---

## Architecture

The system is a **RAG pipeline** — Retrieval-Augmented Generation. Instead of relying on the LLM's training data alone, it first searches a local vector knowledge base (the indexed GitLab handbook), retrieves the most relevant chunks, then uses those as grounding context to generate an answer. This eliminates hallucination of GitLab-specific facts and keeps answers up-to-date with the indexed pages.

### Two-phase design

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PHASE 1 — Index build  (runs once, ~60 seconds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  GitLab URLs
      │
      ▼
  WebBaseLoader          scrapes HTML, strips tags → plain text Documents
      │
      ▼
  RecursiveCharacterTextSplitter   splits into 1500-char overlapping chunks
      │
      ▼
  HuggingFace Embeddings           converts each chunk → 384-dim float vector
  (all-MiniLM-L6-v2)
      │
      ▼
  FAISS.from_documents()           builds vector index in memory
      │
      ▼
  faiss_index/ on disk             persisted — skipped on all future runs


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PHASE 2 — Per-message query  (runs in ~1-2 seconds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  User types question
      │
      ▼
  RunnableWithMessageHistory       injects chat_history into the chain
      │
      ▼
  create_history_aware_retriever   if history exists:
      │                              LLM rewrites follow-up → standalone question
      │                            if no history:
      │                              question passed through as-is
      ▼
  FAISS MMR search                 standalone question → top 5 diverse chunks
      │
      ▼
  create_stuff_documents_chain     chunks + chat_history + question → LLM prompt
      │
      ▼
  Groq (Llama 3.3 70B)            generates grounded answer
      │
      ▼
  Streamlit UI                     renders answer bubble + source chips
      │
      ▼
  InMemoryChatMessageHistory       saves [HumanMessage, AIMessage] to session
```

---

## Tech stack & why

| Component | Choice | Reason |
|---|---|---|
| **UI framework** | Streamlit | Native `st.chat_input`, session state, one-command cloud deployment |
| **LLM** | Groq — Llama 3.3 70B Versatile | Extremely fast inference (~500 tok/s), generous free tier, plain string output (no thinking tokens that break history) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Runs fully locally — no API key, no rate limits, no cost, downloads once and caches |
| **Vector store** | FAISS (local) | Zero external dependencies, persists to two small files, fast enough for this scale |
| **Retrieval** | MMR search (k=5, fetch_k=20) | Maximal Marginal Relevance returns diverse chunks, not 5 copies of the same paragraph |
| **Chain** | `create_history_aware_retriever` + `create_retrieval_chain` | Modern LangChain LCEL — handles question condensing + multi-turn correctly |
| **Memory** | `RunnableWithMessageHistory` + `InMemoryChatMessageHistory` | Per-session memory stored in `st.session_state`, survives Streamlit reruns |
| **Scraping** | `WebBaseLoader` | Built-in BeautifulSoup scraper, returns clean text + source URL metadata |

### Why Groq over Gemini / OpenAI?

Groq runs LLMs on custom LPU (Language Processing Unit) hardware, making it 5–10x faster than GPU-based APIs. The free tier is generous, and critically — Groq models return `AIMessage.content` as a **plain string** always. This matters because Gemini 2.5+ returns thinking tokens as a list of content blocks, which breaks `InMemoryChatMessageHistory` and require complex token handling mechanisms, makin it prone to errors. Groq has none of this complexity.

### Why HuggingFace embeddings over Google/OpenAI?

`all-MiniLM-L6-v2` runs locally — the model downloads on first use and is cached permanently. No embedding API calls means no rate limits during index building, no cost per token, and no external dependency that can go down.

---


## Quick start

### Prerequisites

- Python 3.9+
- A free [Groq API key](https://console.groq.com) — takes 30 seconds to create, starts with `gsk_`

### 1. Clone and install

```bash
git clone https://github.com/Parth-1809/Gitlab-Employee
cd Gitlab-Employee-Chatbot

venv\Scripts\activate
source venv/bin/activate        

pip install -r requirements.txt
```

The first `pip install` also downloads the `all-MiniLM-L6-v2` embedding model (~90MB) into HuggingFace's local cache. This only happens once.

### 2. Set your API key

```bash
cp .env.example .env
```

Open `.env` and add your Groq key:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

Get a free key at [console.groq.com](https://console.groq.com) → API Keys → Create new key.

### 3. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Build the index

On first launch, click **"Build / Refresh Index"** in the sidebar. This will:

1. Scrape the configured GitLab handbook URLs
2. Split the content into overlapping chunks
3. Embed every chunk locally using HuggingFace
4. Save the FAISS index to `./faiss_index/`

Takes about 1-2 minutes depending on your internet connection. Every subsequent run loads from disk in under a second — no re-scraping or re-embedding needed.

### 5. Start chatting

Once the sidebar shows **"Index ready"**, the chat input unlocks. Try:

- _"What are GitLab's core values?"_
- _"How does async communication work?"_
- _"Tell me more about the first one"_ ← follow-up, uses conversation memory

---

## How the index works

### Scraping

`WebBaseLoader` fetches each URL using `requests`, parses the HTML with BeautifulSoup, strips all tags, and returns clean plain text. Each page becomes a `Document` object carrying the source URL as metadata — this is what powers the source chips shown under each answer.

### Chunking

`RecursiveCharacterTextSplitter` splits documents with `chunk_size=1500` and `chunk_overlap=100`. It tries to split on `\n\n` (paragraphs) first, then `\n` (lines), then `. ` (sentences) — preserving semantic boundaries wherever possible. The 100-character overlap means sentences near a chunk boundary appear in both adjacent chunks, preventing context loss.

### Embedding

Each chunk is converted to a 384-dimensional float vector by `all-MiniLM-L6-v2`. Semantically similar text gets similar vectors — this is what makes the search understand meaning rather than just keywords.

### Retrieval (MMR)

At query time, the question is embedded and compared against all stored vectors. MMR (Maximal Marginal Relevance) fetches 20 candidate chunks but returns only the 5 that are most relevant **and** maximally diverse from each other. This prevents the LLM from receiving 5 near-identical excerpts from the same paragraph when a topic is mentioned repeatedly.

### Disk persistence

`FAISS.save_local()` writes two files: `index.faiss` (binary vectors) and `index.pkl` (chunk text + metadata). `@st.cache_resource` loads these into memory once per server process and shares them across all browser sessions. Rebuilding deletes both files and re-runs the full pipeline.

---

## How conversation memory works

Two memory systems run in parallel — both must be kept in sync:

### 1. `st.session_state.messages` (UI layer)

A plain list of `{"role": "user"|"assistant", "content": str, "sources": [...]}` dicts. Streamlit reruns the entire script on every interaction — `session_state` is the only storage that survives reruns. This list drives the chat bubble rendering loop.

### 2. `InMemoryChatMessageHistory` (LangChain layer)

Stores `HumanMessage` and `AIMessage` objects per session UUID. `RunnableWithMessageHistory` automatically reads from and writes to this store on every `chain.invoke()` call. It injects the history into both:

- The **contextualize prompt** — so the question condenser can rewrite follow-ups into standalone queries
- The **QA prompt** — so the answer generator can reference previous turns directly

### Why both are needed

`session_state.messages` is for rendering; `InMemoryChatMessageHistory` is for the chain. Clearing only one causes a split-brain — the UI looks empty but the LLM still has context from old turns, or vice versa. The Clear Chat button resets both simultaneously and generates a fresh `session_id`.

### The question condenser

This is the key to multi-turn working correctly. Without it:

- Turn 1: _"What are GitLab's values?"_ → retrieves CREDIT acronym chunks ✅
- Turn 2: _"Tell me more about the Collaboration"_ → retrieves nothing useful ❌

With `create_history_aware_retriever`, Turn 2 becomes:

1. LLM receives: history + _"Tell me more about the Collaboration"_
2. LLM outputs: _"Tell me more about the Collaboration value in GitLab"_
3. **This condensed question** goes to FAISS → correct chunks retrieved ✅

---

## Adding more pages

Edit `GITLAB_URLS` in `app.py`:

```python
GITLAB_URLS = [
    "https://about.gitlab.com/handbook/",
    "https://about.gitlab.com/whats-new/#whats-coming",
    Add any url here
]

```

Then click **"Build / Refresh Index"** in the sidebar to re-scrape and re-embed. Be mindful that each additional page increases index build time and the `faiss_index/` folder size proportionally.

---

## Guardrails & transparency

| Guardrail | Implementation |
|---|---|
| **Source attribution** | Every answer shows clickable chips linking to the exact handbook pages that were retrieved |
| **Grounded prompting** | System prompt says "Use ONLY the following context" and instructs the model to say "I don't have that information" rather than speculate |
| **Low temperature (0.3)** | Reduces randomness — answers are consistent and factual rather than creative |
| **MMR diversity** | Retrieved chunks are diverse, reducing the risk of one-sided or repetitive answers |
| **"I don't know" path** | When the answer isn't in the indexed pages, the bot explicitly directs users to `handbook.gitlab.com` rather than hallucinating |
