import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from pdf_processor import extract_pdf_pages
from chunker import chunk_pages
from embeddings import load_embedding_model, embed_chunks
from retriever import build_index, retrieve_chunks
from answer_generator import generate_answer


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Cognidoc AI",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------
# STYLING — FULL-PAGE CHAT INTERFACE
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0c0c12;
        --panel: #15151f;
        --panel-soft: #1b1b28;
        --border: #2d2d3d;
        --text: #f4f2fb;
        --muted: #a09fb4;
        --accent: #a78bfa;
        --accent-strong: #8b5cf6;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 50% -15%, rgba(124, 58, 237, 0.15), transparent 42%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(12, 12, 18, 0.85);
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    .block-container {
        max-width: 900px;
        padding-top: 1.5rem;
        padding-bottom: 7rem;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.5px;
        color: var(--text) !important;
    }

    p, li, label {
        color: #d0cede;
    }

    .topbar {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 5px 0 20px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 28px;
    }

    .brand-mark {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        flex-shrink: 0;
        border-radius: 13px;
        background: linear-gradient(135deg, #b6a0ff, #7c3aed);
        color: white;
        font-size: 23px;
        box-shadow: 0 5px 24px rgba(139, 92, 246, 0.24);
    }

    .brand-name {
        color: #f7f4ff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.6px;
        line-height: 1.2;
    }

    .brand-subtitle {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 4px;
    }

    .topbar-right {
        margin-left: auto;
        color: #b7a5ff;
        border: 1px solid #39304f;
        background: rgba(139, 92, 246, 0.08);
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 0.72rem;
        white-space: nowrap;
    }

    .hero {
        text-align: center;
        padding: 28px 0 22px 0;
    }

    .hero-icon {
        width: 58px;
        height: 58px;
        margin: 0 auto 17px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid #40345e;
        border-radius: 19px;
        background: linear-gradient(145deg, #29213f, #171321);
        color: #c4b5fd;
        font-size: 28px;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.9rem, 5vw, 2.7rem);
        line-height: 1.15;
        font-weight: 700;
        letter-spacing: -1.5px;
        color: #f7f5ff;
        margin-bottom: 12px;
    }

    .hero-title span {
        background: linear-gradient(90deg, #d0c3ff, #9b75ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-copy {
        max-width: 590px;
        margin: 0 auto;
        color: #a6a4b9;
        font-size: 0.96rem;
        line-height: 1.75;
    }

    .upload-card {
        border: 1px dashed #51466f;
        border-radius: 17px;
        padding: 18px 20px 12px 20px;
        background: rgba(24, 22, 35, 0.75);
        margin: 14px 0 18px 0;
    }

    .upload-title {
        color: #f1edff;
        font-weight: 600;
        font-size: 0.98rem;
        margin-bottom: 5px;
    }

    .upload-help {
        color: #9c9ab0;
        font-size: 0.82rem;
        margin-bottom: 12px;
    }

    [data-testid="stFileUploader"] {
        background: transparent;
        border: none;
        padding: 0;
    }

    [data-testid="stFileUploader"] section {
        background: rgba(255,255,255,0.02);
        border: 1px solid #353247;
        border-radius: 12px;
    }

    [data-testid="stFileUploader"] small {
        color: #a6a4b9 !important;
    }

    .document-chip {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 12px 14px;
        border: 1px solid #39334c;
        background: rgba(139, 92, 246, 0.08);
        border-radius: 13px;
        margin: 10px 0 18px 0;
    }

    .document-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: #2b2342;
        color: #c4b5fd;
        font-size: 18px;
        flex-shrink: 0;
    }

    .document-name {
        color: #f0edfa;
        font-weight: 600;
        overflow-wrap: anywhere;
    }

    .document-meta {
        color: #a09fb4;
        font-size: 0.77rem;
        margin-top: 3px;
    }

    .section-kicker {
        color: #9b8bc9;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 20px 0 10px 0;
    }

    .suggestion-note {
        color: #aaa8bc;
        font-size: 0.86rem;
        margin: 3px 0 11px 0;
    }

    .stButton > button {
        border: 1px solid #363348;
        background: #171621;
        color: #ded9f0;
        border-radius: 12px;
        min-height: 44px;
        text-align: left;
        transition: all 0.18s ease;
    }

    .stButton > button:hover {
        border-color: #8b5cf6;
        background: #211b31;
        color: white;
    }

    [data-testid="stChatMessage"] {
        border: 1px solid #292938;
        border-radius: 17px;
        padding: 14px 17px;
        margin-bottom: 14px;
        background: rgba(21, 21, 31, 0.78);
    }

    [data-testid="stChatMessage"] p {
        line-height: 1.75;
    }

    [data-testid="stChatInput"] {
        border-color: #49405f !important;
        background: #171621 !important;
        border-radius: 17px !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #f4f2fb !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #8e8ba2 !important;
    }

    [data-testid="stExpander"] {
        background: #14141e;
        border: 1px solid #302d40;
        border-radius: 13px;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    .empty-state {
        text-align: center;
        color: #8e8ba2;
        font-size: 0.86rem;
        padding: 22px 12px;
        border: 1px solid #292938;
        background: rgba(21, 21, 31, 0.5);
        border-radius: 15px;
        margin: 16px 0;
    }

    .footer {
        color: #77758b;
        font-size: 0.75rem;
        text-align: center;
        padding-top: 34px;
        line-height: 1.8;
    }

    hr {
        border-color: #292938;
    }

    @media (max-width: 600px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }

        .topbar-right {
            display: none;
        }

        .hero {
            padding-top: 15px;
        }
    }

    /* Cognidoc glassmorphism visual refresh */
    .stApp {
        background:
            radial-gradient(ellipse at 8% 8%, rgba(117, 55, 220, 0.22), transparent 28%),
            radial-gradient(ellipse at 92% 80%, rgba(58, 100, 235, 0.17), transparent 27%),
            radial-gradient(ellipse at 52% 0%, rgba(48, 95, 170, 0.13), transparent 30%),
            #080a18;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 1.2rem;
        padding-bottom: 4rem;
    }

    .topbar {
        display: block;
        text-align: center;
        border: 1px solid rgba(166, 133, 255, 0.45);
        border-radius: 30px;
        padding: 28px 30px 25px;
        margin: 0 auto 22px;
        background:
            radial-gradient(ellipse at 50% 35%, rgba(110, 64, 205, 0.22), transparent 48%),
            linear-gradient(135deg, rgba(28, 27, 55, 0.88), rgba(14, 18, 39, 0.92));
        box-shadow: 0 0 35px rgba(112, 66, 235, 0.12), inset 0 1px 0 rgba(255,255,255,0.07);
        position: relative;
    }

    .brand-mark {
        display: none;
    }

    .brand-name {
        font-size: clamp(2.5rem, 6vw, 3.5rem);
        line-height: 1.1;
        letter-spacing: -1.8px;
        font-weight: 700;
    }

    .brand-name::after {
        content: " AI";
        color: #a77bff;
    }

    .brand-subtitle {
        font-size: 1.1rem;
        color: #c2c0d7;
        margin-top: 9px;
    }

    .topbar-right {
        display: inline-block;
        margin: 13px 0 0;
        border: none;
        background: transparent;
        color: #bcb3d7;
        letter-spacing: 4px;
        font-size: 0.68rem;
        font-weight: 700;
    }

    .hero {
        padding: 8px 0 4px;
    }

    .hero-icon {
        display: none;
    }

    .hero-title {
        display: none;
    }

    .hero-copy {
        display: none;
    }

    .upload-card {
        border: 1px dashed rgba(169, 143, 255, 0.48);
        background: rgba(31, 31, 60, 0.52);
        border-radius: 17px;
        padding: 12px 16px 8px;
        margin: 10px 0 18px;
    }

    [data-testid="stFileUploader"] {
        background: rgba(31, 31, 60, 0.68);
        border: 1px dashed rgba(166, 133, 255, 0.5);
        border-radius: 16px;
        padding: 8px 12px;
        margin: 0 0 14px;
    }

    [data-testid="stFileUploader"] section {
        background: transparent;
        border: none;
    }

    [data-testid="stFileUploader"] button {
        border-radius: 10px !important;
        background: linear-gradient(135deg, #8b5cf6, #6d4aff) !important;
        color: white !important;
        border: 0 !important;
    }

    .document-chip {
        display: inline-flex;
        max-width: 100%;
        margin: 6px 0 12px;
        background: rgba(101, 82, 156, 0.2);
        border: 1px solid rgba(177, 151, 255, 0.35);
        border-radius: 999px;
        padding: 9px 15px;
    }

    .document-icon {
        width: 32px;
        height: 32px;
        border-radius: 9px;
    }

    .stButton > button {
        border: 1px solid rgba(158, 144, 207, 0.28);
        background: linear-gradient(135deg, rgba(39, 37, 67, 0.9), rgba(26, 28, 52, 0.92));
        color: #f1edff;
        border-radius: 14px;
        min-height: 42px;
        text-align: center;
        font-size: 0.85rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }

    .stButton > button:hover {
        border-color: #a78bfa;
        background: rgba(105, 75, 176, 0.35);
        color: white;
    }

    [data-testid="stChatInput"] {
        border: 1px solid rgba(174, 157, 230, 0.6) !important;
        background: linear-gradient(120deg, rgba(36, 34, 61, 0.97), rgba(24, 28, 54, 0.97)) !important;
        border-radius: 19px !important;
        box-shadow: 0 8px 30px rgba(4, 5, 20, 0.25);
    }

    [data-testid="stChatInput"] textarea {
        color: #f7f4ff !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #aaa5c2 !important;
    }

    [data-testid="stChatMessage"] {
        border: 1px solid rgba(150, 134, 207, 0.24);
        border-radius: 17px;
        background: rgba(19, 21, 43, 0.84);
        box-shadow: 0 8px 22px rgba(0,0,0,0.12);
    }

    [data-testid="stExpander"] {
        background: rgba(19, 21, 43, 0.82);
        border: 1px solid rgba(150, 134, 207, 0.28);
        border-radius: 14px;
    }

    .footer {
        color: #797993;
        padding-top: 24px;
    }

    @media (max-width: 600px) {
        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }

        .topbar {
            padding: 22px 15px;
            border-radius: 22px;
        }

        .brand-name {
            font-size: 2.35rem;
        }

        .brand-subtitle {
            font-size: 0.95rem;
        }

        .topbar-right {
            letter-spacing: 2px;
            font-size: 0.6rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

DEFAULTS = {
    "file_hash": None,
    "temporary_pdf_path": None,
    "chunks": None,
    "index": None,
    "model": None,
    "page_count": 0,
    "chunk_count": 0,
    "file_name": None,
    "messages": [],
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# PDF PROCESSING
# --------------------------------------------------

def process_pdf(uploaded_file):
    """Extract, chunk, embed, and index an uploaded PDF."""
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.get("file_hash") == file_hash:
        return

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary_file:
            temporary_file.write(file_bytes)
            temporary_path = Path(temporary_file.name)

        pages = extract_pdf_pages(temporary_path)

        if not pages:
            raise ValueError("No pages could be extracted from this PDF.")

        for page in pages:
            page["source"] = uploaded_file.name

        chunks = chunk_pages(pages, chunk_size=700, overlap=120)

        if not chunks:
            raise ValueError("No searchable text could be extracted from this PDF.")

        model = get_embedding_model()
        embeddings = embed_chunks(chunks, model)
        index = build_index(embeddings)

        old_path = st.session_state.get("temporary_pdf_path")
        if old_path:
            Path(old_path).unlink(missing_ok=True)

        st.session_state["file_hash"] = file_hash
        st.session_state["temporary_pdf_path"] = str(temporary_path)
        st.session_state["chunks"] = chunks
        st.session_state["index"] = index
        st.session_state["model"] = model
        st.session_state["page_count"] = len(pages)
        st.session_state["chunk_count"] = len(chunks)
        st.session_state["file_name"] = uploaded_file.name
        st.session_state["messages"] = []

        temporary_path = None

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def clear_document():
    """Clear the current document and its conversation."""
    old_path = st.session_state.get("temporary_pdf_path")
    if old_path:
        Path(old_path).unlink(missing_ok=True)

    for key, value in DEFAULTS.items():
        st.session_state[key] = value

    st.session_state["uploaded_pdf"] = None


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="topbar">
        <div>
            <div class="brand-name">Cognidoc</div>
            <div class="brand-subtitle">Chat with your document</div>
            <div class="topbar-right">✦ &nbsp; AI DOCUMENT ASSISTANT</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# UPLOAD AREA
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    key="uploaded_pdf",
    help="Choose a text-based PDF to ask questions about.",
    label_visibility="collapsed",
)

if not st.session_state.get("chunks"):
    st.markdown(
        """
        <div class="upload-card">
            <div class="upload-title">＋ &nbsp; Upload your PDF</div>
            <div class="upload-help">Add a document to start chatting with it.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if uploaded_file is not None:
    try:
        uploaded_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()

        if st.session_state.get("file_hash") != uploaded_hash:
            with st.spinner("Reading your PDF and preparing it for questions..."):
                process_pdf(uploaded_file)
            st.success("Your document is ready. Ask your first question below.")

    except Exception as error:
        st.error(
            "We couldn't process this PDF. Make sure it is a valid, unencrypted PDF "
            "with extractable text."
        )
        print(f"PDF processing error: {error}")


# --------------------------------------------------
# DOCUMENT READY: CHAT INTERFACE
# --------------------------------------------------

if st.session_state.get("chunks"):
    st.markdown(
        f"""
        <div class="document-chip">
            <div class="document-icon">▤</div>
            <div style="min-width:0; flex:1;">
                <div class="document-name">{st.session_state["file_name"]}</div>
                <div class="document-meta">
                    {st.session_state["page_count"]} pages
                    &nbsp;·&nbsp; {st.session_state["chunk_count"]} searchable passages
                    &nbsp;·&nbsp; Indexed and ready
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    action_col, info_col = st.columns([1, 2])

    with action_col:
        if st.button("✦  New conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

    with info_col:
        with st.popover("▤  Document details"):
            st.write(f"**File:** {st.session_state['file_name']}")
            st.write(f"**Pages:** {st.session_state['page_count']}")
            st.write(f"**Searchable passages:** {st.session_state['chunk_count']}")
            st.caption(
                "Upload another PDF above to replace the current document and start a new conversation."
            )

    if not st.session_state["messages"]:
        st.markdown(
            '<div class="section-kicker">GET STARTED</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="suggestion-note">Choose a suggestion or type your own question below.</div>',
            unsafe_allow_html=True,
        )

        suggestions = [
            "📄  Summarize document",
            "⌕  Find key details",
            "✦  Explain this topic",
        ]

        suggestion_cols = st.columns(3)
        for column, suggestion in zip(suggestion_cols, suggestions):
            with column:
                if st.button(suggestion, key=f"suggest_{suggestion}", use_container_width=True):
                    st.session_state["pending_question"] = suggestion
                    st.rerun()

    # Render existing conversation.
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("View supporting passages"):
                    for rank, result in enumerate(message["sources"], start=1):
                        st.markdown(
                            f"**Source {rank} · {result.get('source', st.session_state['file_name'])} "
                            f"· Page {result.get('page', '?')}**"
                        )
                        st.caption(f"Retrieval similarity: {result.get('score', 0):.4f}")
                        st.write(result.get("text", ""))
                        if rank < len(message["sources"]):
                            st.divider()

    pending_question = st.session_state.pop("pending_question", None)
    question = st.chat_input("Ask anything about your PDF...")

    if pending_question and not question:
        question = pending_question

    quick_question_map = {
        "📄  Summarize document": "Summarize this document and its key findings.",
        "⌕  Find key details": "What are the key details in this document?",
        "✦  Explain this topic": "Explain the main topic of this document.",
    }

    if question:
        question = quick_question_map.get(question.strip(), question.strip())

        if question:
            st.session_state["messages"].append(
                {"role": "user", "content": question}
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching your document..."):
                    try:
                        results = retrieve_chunks(
                            question=question,
                            chunks=st.session_state["chunks"],
                            index=st.session_state["index"],
                            model=st.session_state["model"],
                            top_k=5,
                        )

                        if not results:
                            answer = (
                                "I couldn't find relevant information in this document "
                                "to answer that question. Try rephrasing it or asking about "
                                "a topic covered in the PDF."
                            )
                        else:
                            answer = generate_answer(
                                question=question,
                                retrieved_chunks=results,
                            )

                        st.markdown(answer)

                        if results:
                            with st.expander("View supporting passages"):
                                for rank, result in enumerate(results, start=1):
                                    st.markdown(
                                        f"**Source {rank} · {result.get('source', st.session_state['file_name'])} "
                                        f"· Page {result.get('page', '?')}**"
                                    )
                                    
                                    st.caption(
    f"Semantic similarity: {result.get('score', 0):.4f} "
    f"| Keyword relevance: {result.get('keyword_score', 0):.4f} "
    f"| Combined rank: {result.get('rerank_score', 0):.4f}"
)
                                    
                                    st.write(result.get("text", ""))
                                    if rank < len(results):
                                        st.divider()

                        st.session_state["messages"].append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": results,
                            }
                        )

                    except Exception as error:
                        error_message = (
                            "I couldn't generate an answer just now. Please check your "
                            "API configuration and try again."
                        )
                        st.error(error_message)
                        st.session_state["messages"].append(
                            {
                                "role": "assistant",
                                "content": error_message,
                                "sources": [],
                            }
                        )
                        print(f"Answer generation error: {error}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        COGNIDOC AI · DOCUMENT INTELLIGENCE<br>
        Answers are generated from retrieved passages. Verify important claims against the original PDF.
    </div>
    """,
    unsafe_allow_html=True,
)
