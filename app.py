import streamlit as st
import os
import time
import traceback
import uuid

from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FAISS_INDEX_PATH = "faiss_index"

GITLAB_URLS = [
    "https://about.gitlab.com/handbook/",
    "https://about.gitlab.com/whats-new/#whats-coming",
    #To do: We can add more URLs if required, being mindful of the total token.
]


st.set_page_config(
    page_title="GitLab Handbook Assistant",
    page_icon="https://about.gitlab.com/images/press/logo/png/gitlab-icon-rgb.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

#Styling for the app
def load_css(file_path: str):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


@st.cache_resource(show_spinner=False)
def load_or_build_index(api_key: str):
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if Path(FAISS_INDEX_PATH).exists():
        db = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20},
        )
    

    try:
        loader = WebBaseLoader(GITLAB_URLS)
        loader.requests_kwargs = {"timeout": 30}

        raw_docs = loader.load()

    except Exception as e:
        error_type = type(e).__name__;
        error_message = str(e);
        full_traceback = traceback.format_exc();
        st.error(f"Error during scraping: {error_type}: {error_message}\n\n{full_traceback}")

    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.split_documents(raw_docs)

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(FAISS_INDEX_PATH)

    return db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20},
    )

def build_chain(api_key: str, retriever):
    llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3,
    )

    system_prompt = (
    "You are a helpful assistant for GitLab employees and candidates and questions will be asked about Gitlab."
    "Use ONLY the following context from the GitLab Handbook and Direction pages to answer the question."
    "If the answer is not in the context, say:"
    "I don't have that information in the handbook sections I've indexed — try checking https://handbook.gitlab.com directly."
    "Be concise, accurate, and friendly. Use bullet points where helpful."
    "If quoting specific policies, mention which handbook section it's from."
    "\n\n"
    "{context}"
    )

    contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )

    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )   

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=contextualize_q_prompt,
    )

    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt,
    )

    chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain,
    )

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

#Session state initialization.
if "messages" not in st.session_state:
    st.session_state.messages = []  

if "chain" not in st.session_state:
    st.session_state.chain = None

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[session_id] = InMemoryChatMessageHistory()
    return st.session_state.chat_histories[session_id]

with st.sidebar:
    st.markdown("## GitLab Assistant")
    st.markdown("---")

    api_key_input = GROQ_API_KEY

    if st.button(" Build / Refresh Index", use_container_width=True):
        if not api_key_input:
            st.error("Please enter your GROQ API Key first.")
        else:
            with st.spinner("Scraping GitLab pages & building vector index…\nThis takes ~60s the first time."):
                load_or_build_index.clear()
                if Path(FAISS_INDEX_PATH).exists():
                    import shutil
                    shutil.rmtree(FAISS_INDEX_PATH)
                retriever = load_or_build_index(api_key_input)
                st.session_state.chain = build_chain(api_key_input, retriever)
                st.session_state.index_ready = True
            st.success("Index built and ready!")

    if not st.session_state.index_ready and api_key_input and Path(FAISS_INDEX_PATH).exists():
        retriever = load_or_build_index(api_key_input)
        st.session_state.chain = build_chain(api_key_input, retriever)
        st.session_state.index_ready = True

    st.markdown("---")
    if st.session_state.index_ready:
        st.markdown('<span class="status-badge status-ready">Index ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-loading">Index not loaded</span>', unsafe_allow_html=True)

    st.markdown("---")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = {}
        st.session_state.session_id = str(uuid.uuid4())
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
    st.caption("Built with LangChain · GROQ · FAISS · Streamlit")

chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
<div style="text-align:center; padding: 3rem 1rem; color: #888;">
    <div style="font-size: 3rem;"><img src="https://about.gitlab.com/images/press/logo/png/gitlab-icon-rgb.png"></div>
    <div style="font-size: 1.1rem; margin-top: 0.5rem; font-weight: 500;">Ask me anything about GitLab</div>
    <div style="font-size: 0.85rem; margin-top: 0.5rem;">
        Try: <em>"What is GitLab's remote work culture?"</em> or <em>"What is Gitlab's Flexible schedules?"</em>
    </div>
</div>
""", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        avatar = (
        "https://www.iconpacks.net/icons/5/free-no-profile-picture-icon-15258-thumb.png"
         if role == "user"
         else
        "https://avatars.githubusercontent.com/u/1086321?s=280&v=4"
         )
        css_class = "user" if role == "user" else "bot"

        st.markdown(f"""
<div class="chat-message {css_class}">
    <div class="avatar">
    <img src="{avatar}" />
    </div>
    <div class="bubble">{content}</div>
</div>
""", unsafe_allow_html=True)

        if role == "assistant" and msg.get("sources"):
            unique_sources = list(dict.fromkeys(msg["sources"]))
            chips = "".join(f'<a href="{s}" target="_blank" class="source-chip">📄 {s.split("/")[-2] or s.split("/")[-1]}</a>' for s in unique_sources[:4])
            st.markdown(f'<div style="margin-left:48px; margin-bottom:0.5rem;">{chips}</div>', unsafe_allow_html=True)

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
                history = get_session_history(st.session_state.session_id)
                result = st.session_state.chain.invoke(
                    {"input": user_input},
                    {"configurable": {"session_id": st.session_state.session_id}},
                )
                answer = result["answer"]
                source_docs = result.get("context", [])
                sources = [doc.metadata.get("source", "") for doc in source_docs if doc.metadata.get("source")]
            except Exception as e:
                error_type = type(e).__name__;
                error_message = str(e);
                full_traceback = traceback.format_exc();
                st.error(f"Error during scraping: {error_type}: {error_message}\n\n{full_traceback}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        st.rerun()
