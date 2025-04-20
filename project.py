import os
import tempfile
import pickle

import numpy as np
import fitz  # PyMuPDF
import faiss
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai

# ── Page config & styling ─────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄", layout="wide")
st.markdown("""
    <style>
      body { background-color: #f0f2f6; }
      .header h1 { color:#4B4E6D; font-size:2.8rem; margin-bottom:5px; }
      .header p  { color:#6C6F7D; font-size:1.1rem; }
      .chat-container { max-height:600px; overflow-y:auto; padding:12px; background:#fff; border-radius:10px; border:1px solid #e1e4e8; }
      .user-msg {
        background:#E8F0FE;
        color: #000;          /* make all text black */
        padding:10px 15px;
        border-radius:15px 15px 0 15px;
        margin:8px 0;
        max-width:75%;
      }
      .bot-msg {
        background:#F1F3F4;
        color: #000;          /* make all text black */
        padding:10px 15px;
        border-radius:15px 15px 15px 0;
        margin:8px 0;
        max-width:75%;
        margin-left:auto;
      }
      .sidebar .stButton>button { background-color:#4B4E6D; color:white; }
    </style>
""", unsafe_allow_html=True)

# ── API & GenAI setup ───────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA3rmQ5OxkoR3BMHmEPkesDKSH4Cn7ZrbY")
genai.configure(api_key=GOOGLE_API_KEY)

# ── Session state flags ────────────────────────────────────────────────────────
st.session_state.setdefault("indexed", False)
st.session_state.setdefault("history", [])

# ── Build FAISS index & save chunks ────────────────────────────────────────────
def build_faiss_from_pdf(pdf_bytes: bytes, chunk_size: int = 1000, overlap: int = 200):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        doc = fitz.open(tmp.name)
    text = "".join(page.get_text() for page in doc)
    doc.close()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(text)
    if not chunks:
        st.error("❌ No text extracted from PDF.")
        return

    embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vectors = embedder.embed_documents(chunks)
    arr = np.array(vectors, dtype="float32")

    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)
    faiss.write_index(index, "faiss_index.idx")

    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    st.session_state.indexed = True
    st.success("✅ PDF indexed and embeddings stored!")

# ── Load FAISS index & chunks ──────────────────────────────────────────────────
def load_index_and_chunks():
    index = faiss.read_index("faiss_index.idx")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# ── QA chain setup ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_qa_chain():
    template = """
Answer from the provided context. If not found, reply:
"answer is not available in the context."

Context:
{context}

Question:
{question}

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                   temperature=0.3,
                                   google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=template,
                            input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def answer_question(question: str, k: int = 5) -> str:
    index, chunks = load_index_and_chunks()
    embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    q_vec = np.array(embedder.embed_query(question), dtype="float32")[None, :]
    D, I = index.search(q_vec, k)
    docs = [Document(page_content=chunks[i]) for i in I[0] if i < len(chunks)]

    chain = get_qa_chain()
    out = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return out["output_text"]

# ── Sidebar UI: upload & index ────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload & Index PDF")
    uploaded_file = st.file_uploader("Select a PDF", type="pdf")
    if uploaded_file and st.button("Process PDF"):
        with st.spinner("Indexing large PDF…"):
            build_faiss_from_pdf(uploaded_file.read())
    st.markdown("---")
    st.caption("Built by Deven Sharma")

# ── Main header & chat UI ─────────────────────────────────────────────────────
st.markdown("""
    <div class="header">
      <h1>📄 PDF Q&A Chatbot</h1>
      <p>Ask questions and get context‑aware answers from your PDF.</p>
    </div>
""", unsafe_allow_html=True)

# Render history
# st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for turn in st.session_state.history:
    cls = "user-msg" if turn["role"] == "user" else "bot-msg"
    label = "You" if turn["role"] == "user" else "Chatbot"
    st.markdown(
        f'<div class="{cls}"><strong>{label}:</strong> {turn["text"]}</div>',
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)

# Chat input (disabled until indexed)
if st.session_state.indexed:
    with st.form("ask_form", clear_on_submit=True):
        q = st.text_input("Your question…", placeholder="")
        submit = st.form_submit_button("Send")
        if submit and q:
            st.session_state.history.append({"role": "user", "text": q})
            with st.spinner("Thinking…"):
                ans = answer_question(q)
            st.session_state.history.append({"role": "assistant", "text": ans})
            st.rerun()
else:
    st.info("🔒 Please upload & process a PDF first to start chatting.")