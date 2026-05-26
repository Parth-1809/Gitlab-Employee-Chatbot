"""
GitLab Handbook Chatbot
=======================
Stack: Streamlit + LangChain + Gemini API + FAISS

Key design decisions:
- ConversationalRetrievalChain: automatically condenses follow-up questions
  into standalone queries using chat history before hitting the retriever.
  Without this, "What does it say about that?" would retrieve nothing useful.
- FAISS with local persistence: index is built once and saved to disk.
  Subsequent runs skip the expensive scrape+embed step entirely.
- session_state for history: Streamlit reruns the whole script on every
  interaction; session_state is the only way to persist data across reruns.
"""

import streamlit as st
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# ── Configuration ──────────────────────────────────────────────────────────────

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FAISS_INDEX_PATH = "faiss_index"

# Pages to scrape — GitLab Handbook + Direction
# We pick representative, stable URLs. More can be added freely.
GITLAB_URLS = [
    "https://about.gitlab.com/handbook/",
]

# ── Streamlit page config ──────────────────────────────────────────────────────
# Wide layout gives the chat room to breathe; icon makes browser tab recognisable.
st.set_page_config(
    page_title="GitLab Handbook Assistant",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
# Streamlit's default theme is fine but we polish it to feel more "GitLab":
# orange accent, clean monospace for code, tight message bubbles.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #FC6D26 0%, #E24329 50%, #FCA326 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .header-banner h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .header-banner p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    /* Chat messages */
    .chat-message {
        display: flex;
        gap: 0.75rem;
        margin-bottom: 1rem;
        animation: fadeSlideIn 0.3s ease-out;
    }
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .chat-message .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .chat-message.user .avatar   { background: #FC6D26; }
    .chat-message.bot  .avatar   { background: #2B2D3B; }
    .chat-message .bubble {
        padding: 0.75rem 1rem;
        border-radius: 12px;
        line-height: 1.6;
        max-width: 85%;
        font-size: 0.95rem;
    }
    .chat-message.user .bubble {
        background: #FFF3EE;
        border: 1px solid #FDDCCC;
        color: #2B2D3B;
        margin-left: auto;
    }
    .chat-message.bot .bubble {
        background: #F8F9FA;
        border: 1px solid #E9ECEF;
        color: #2B2D3B;
    }

    /* Source chips */
    .source-chip {
        display: inline-block;
        background: #E8F4FD;
        border: 1px solid #BEE3F8;
        color: #2C6B9E;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 2px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Index status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .status-ready   { background: #D4EDDA; color: #155724; }
    .status-loading { background: #FFF3CD; color: #856404; }

    /* Input area */
    .stTextInput input {
        border-radius: 24px !important;
        border: 2px solid #E9ECEF !important;
        padding: 0.6rem 1.2rem !important;
        font-size: 0.95rem !important;
        transition: border-color 0.2s;
    }
    .stTextInput input:focus {
        border-color: #FC6D26 !important;
        box-shadow: 0 0 0 3px rgba(252,109,38,0.15) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #2B2D3B;
    }
    [data-testid="stSidebar"] * {
        color: #E9ECEF !important;
    }
    [data-testid="stSidebar"] .stButton button {
        background: #FC6D26;
        color: white !important;
        border: none;
        border-radius: 8px;
        width: 100%;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: #E24329;
    }

    /* Hide Streamlit default hamburger / footer */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helper: build/load the vector index ───────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_or_build_index(api_key: str):
    """
    Returns a FAISS retriever, either loaded from disk or freshly built.

    Why @st.cache_resource?
    - cache_resource persists across Streamlit reruns AND across users
      (within the same server process). Building the index takes 30-60s
      the first time; caching means every subsequent load is instant.
    - We pass api_key as an arg so the cache key changes if the key changes.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key,
)

    # ── Load from disk if already built ──
    if Path(FAISS_INDEX_PATH).exists():
        db = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,  # safe: we wrote this file
        )
        return db.as_retriever(
            search_type="mmr",          # Maximal Marginal Relevance: balances
            search_kwargs={"k": 5, "fetch_k": 20},  # relevance vs diversity
        )

    # ── Build from scratch ──
    # Step 1: Scrape
    # WebBaseLoader uses urllib + BeautifulSoup under the hood.
    # It strips HTML tags and returns plain text — good enough for RAG.
    try:
        loader = WebBaseLoader(GITLAB_URLS)
        loader.requests_kwargs = {"timeout": 30}

        raw_docs = loader.load()

    except Exception as e:
        raise RuntimeError(f"Failed to load GitLab pages: {str(e)}")

    # Step 2: Split
    # chunk_size=1000 chars ≈ 200-250 tokens — fits comfortably in Gemini's
    # context while keeping chunks semantically coherent.
    # chunk_overlap=150 ensures sentences at chunk boundaries aren't lost.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.split_documents(raw_docs)

    # Step 3: Embed + store
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(FAISS_INDEX_PATH)

    return db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20},
    )


def build_chain(api_key: str, retriever):
    """
    Build the ConversationalRetrievalChain.

    Why ConversationalRetrievalChain over plain RetrievalQA?
    - It has a built-in "question condenser": before retrieving, it sends
      the current question + chat history to the LLM and asks it to rephrase
      the question as a standalone query.
    - Example: history = ["What are GitLab's values?", "CREDIT"]
               new Q   = "Tell me more about the C"
               condensed = "Tell me more about the Collaboration value in GitLab"
      Without condensing, "Tell me more about the C" retrieves garbage.
    """
    llm = ChatGoogleGenerativeAI(
    model="models/gemini-3.5-flash", # Remove -latest if it failed
    google_api_key=api_key,
    version="v1",             # Force the stable API version
    temperature=0.3,
    )

    # Custom prompt: instructs the model to stay grounded in retrieved context
    # and be transparent when it doesn't know something.
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant for GitLab employees and candidates.
Use ONLY the following context from the GitLab Handbook and Direction pages to answer the question.
If the answer is not in the context, say "I don't have that information in the handbook sections I've indexed — try checking https://handbook.gitlab.com directly."

Be concise, accurate, and friendly. Use bullet points where helpful.
If quoting specific policies, mention which handbook section it's from.

Context:
{context}

Question: {question}

Answer:""",
    )

    # ConversationBufferMemory stores the raw chat history.
    # memory_key must match the chain's expected input key.
    # return_messages=True returns LangChain Message objects (not strings),
    # which the chain knows how to serialize into the condenser prompt.
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,   # We show sources in the UI for transparency
        verbose=False,
    )
    return chain


# ── Session state initialisation ──────────────────────────────────────────────
# Streamlit reruns the entire script on every interaction.
# session_state persists across reruns for a given browser session.

if "messages" not in st.session_state:
    st.session_state.messages = []   # List of {"role": "user"|"assistant", "content": str, "sources": [...]}

if "chain" not in st.session_state:
    st.session_state.chain = None

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🦊 GitLab Assistant")
    st.markdown("---")

    # API key input — stored in session_state, never written to disk
    api_key_input = st.text_input(
        "Google API Key",
        type="password",
        value=GOOGLE_API_KEY or "",
        help="Get a free key at https://aistudio.google.com",
        placeholder="AIza...",
    )

    # Index building trigger
    if st.button("🔄 Build / Refresh Index", use_container_width=True):
        if not api_key_input:
            st.error("Please enter your Google API Key first.")
        else:
            with st.spinner("Scraping GitLab pages & building vector index…\nThis takes ~60s the first time."):
                # Clear cached resource so it rebuilds
                load_or_build_index.clear()
                if Path(FAISS_INDEX_PATH).exists():
                    import shutil
                    shutil.rmtree(FAISS_INDEX_PATH)
                retriever = load_or_build_index(api_key_input)
                st.session_state.chain = build_chain(api_key_input, retriever)
                st.session_state.index_ready = True
            st.success("Index built and ready!")

    # Auto-load if index already exists on disk
    if not st.session_state.index_ready and api_key_input and Path(FAISS_INDEX_PATH).exists():
        retriever = load_or_build_index(api_key_input)
        st.session_state.chain = build_chain(api_key_input, retriever)
        st.session_state.index_ready = True

    # Status indicator
    st.markdown("---")
    if st.session_state.index_ready:
        st.markdown('<span class="status-badge status-ready">✅ Index ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-loading">⏳ Index not loaded</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        # Rebuild chain to reset memory too
        if st.session_state.index_ready and api_key_input:
            retriever = load_or_build_index(api_key_input)
            st.session_state.chain = build_chain(api_key_input, retriever)
        st.rerun()

    st.markdown("---")
    st.markdown("""
**Indexed pages:**
- [Values](https://handbook.gitlab.com/handbook/values/)
- [Communication](https://handbook.gitlab.com/handbook/communication/)
- [Leadership](https://handbook.gitlab.com/handbook/leadership/)
- [Engineering](https://handbook.gitlab.com/handbook/engineering/)
- [Product](https://handbook.gitlab.com/handbook/product/)
- [People Group](https://handbook.gitlab.com/handbook/people-group/)
- [Direction](https://about.gitlab.com/direction/)
- [Maturity](https://about.gitlab.com/direction/maturity/)
""")

    st.markdown("---")
    st.caption("Built with LangChain · Gemini · FAISS · Streamlit")


# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-banner">
    <h1>🦊 GitLab Handbook Assistant</h1>
    <p>Ask anything about GitLab's values, processes, engineering culture, or product direction.</p>
</div>
""", unsafe_allow_html=True)

# Render chat history
# We render each message as a custom HTML bubble (not st.chat_message)
# so we have full styling control.
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
<div style="text-align:center; padding: 3rem 1rem; color: #888;">
    <div style="font-size: 3rem;">🦊</div>
    <div style="font-size: 1.1rem; margin-top: 0.5rem; font-weight: 500;">Ask me anything about GitLab</div>
    <div style="font-size: 0.85rem; margin-top: 0.5rem;">
        Try: <em>"What are GitLab's core values?"</em> or <em>"How does remote communication work?"</em>
    </div>
</div>
""", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        avatar = "👤" if role == "user" else "🦊"
        css_class = "user" if role == "user" else "bot"

        st.markdown(f"""
<div class="chat-message {css_class}">
    <div class="avatar">{avatar}</div>
    <div class="bubble">{content}</div>
</div>
""", unsafe_allow_html=True)

        # Show sources for assistant messages
        if role == "assistant" and msg.get("sources"):
            unique_sources = list(dict.fromkeys(msg["sources"]))  # deduplicate, preserve order
            chips = "".join(f'<a href="{s}" target="_blank" class="source-chip">📄 {s.split("/")[-2] or s.split("/")[-1]}</a>' for s in unique_sources[:4])
            st.markdown(f'<div style="margin-left:48px; margin-bottom:0.5rem;">{chips}</div>', unsafe_allow_html=True)


# ── Chat input ────────────────────────────────────────────────────────────────
# st.chat_input is a Streamlit built-in that pins to the bottom of the page
# and triggers a rerun when the user submits.

user_input = st.chat_input(
    "Ask about GitLab's handbook, values, processes…",
    disabled=not st.session_state.index_ready,
)

if user_input:
    if not st.session_state.chain:
        st.error("Please build the index first using the sidebar button.")
    else:
        # Append user message immediately (optimistic UI)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Stream-style response with a spinner
        with st.spinner("Searching handbook…"):
            try:
                result = st.session_state.chain({"question": user_input})
                answer = result["answer"]
                source_docs = result.get("source_documents", [])
                sources = [doc.metadata.get("source", "") for doc in source_docs if doc.metadata.get("source")]
            except Exception as e:
                answer = f"⚠️ Something went wrong: {str(e)}\n\nPlease check your API key and try again."
                sources = []

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        st.rerun()   # Rerun to render the new messages
