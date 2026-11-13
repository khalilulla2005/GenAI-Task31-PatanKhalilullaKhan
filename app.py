from __future__ import annotations

import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st
from dotenv import load_dotenv

from src.rag_pipeline import (
    build_llm,
    build_vector_store,
    create_qa_chain,
    extract_sources,
    load_uploaded_documents,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Conversational PDF Q&A", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embeddings():
    """Create the local embedding model once and reuse it across sessions."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_retrieval_index(documents, embeddings):
    vector_store, chunk_count = build_vector_store(documents, embeddings)
    return vector_store, chunk_count


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

st.title("Conversational PDF Q&A Chatbot")
st.caption("Upload one or more PDFs, ask document-grounded questions, and continue the conversation with memory.")

with st.sidebar:
    st.header("Knowledge base")
    provider = st.radio(
        "LLM provider",
        options=["ollama", "openai"],
        index=0,
        help="Use Ollama for offline local inference or OpenAI for cloud usage.",
    )

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload one or more PDF or UTF-8 text files. Long documents demonstrate chunking and retrieval.",
    )

    if st.button("Build knowledge base", type="primary", use_container_width=True, disabled=not uploaded_files):
        try:
            with st.spinner("Loading and indexing documents..."):
                documents = []
                with TemporaryDirectory(prefix="pdf-chat-") as temp_dir:
                    for index, uploaded_file in enumerate(uploaded_files):
                        filename = Path(uploaded_file.name).name
                        target_path = Path(temp_dir) / f"{index}_{filename}"
                        target_path.write_bytes(uploaded_file.getvalue())
                        loaded = load_uploaded_documents([target_path])
                        for document in loaded:
                            document.metadata["source"] = filename
                        documents.extend(loaded)

                vector_store, chunk_count = build_retrieval_index(
                    documents, get_embeddings()
                )
                llm = build_llm(provider)
                st.session_state.qa_chain = create_qa_chain(vector_store, llm)
                st.session_state.indexed_files = [file.name for file in uploaded_files]
                st.session_state.chunk_count = chunk_count
                st.session_state.chat_history = []

            st.success(f"Indexed {chunk_count} document chunks from {len(uploaded_files)} file(s).")
        except Exception as exc:  # pragma: no cover - surfaced in the UI
            logger.exception("Could not build the document knowledge base.")
            st.error(f"The knowledge base could not be built: {exc}")

    if st.session_state.qa_chain is not None:
        st.success(f"Ready · {st.session_state.chunk_count} chunks indexed")
        with st.expander("Indexed documents"):
            for name in st.session_state.indexed_files:
                st.write(f"• {name}")

    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        st.warning("Set OPENAI_API_KEY in your environment or .env file before selecting OpenAI.")
    if provider == "ollama":
        st.caption(f"Local model: `{os.getenv('OLLAMA_MODEL', 'llama3.1')}`")

    if st.button("Clear chat history", use_container_width=True):
        if st.session_state.qa_chain is not None and st.session_state.qa_chain.memory is not None:
            st.session_state.qa_chain.memory.clear()
        st.session_state.chat_history = []
        st.rerun()

for message in st.session_state.chat_history:
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        st.markdown(content)
        if message.get("sources"):
            st.caption("Sources: " + " · ".join(message["sources"]))

question = st.chat_input("Ask a question about the uploaded PDFs...", disabled=st.session_state.qa_chain is None)

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.qa_chain is None:
        st.warning("Build the knowledge base before asking questions.")
    else:
        try:
            result = st.session_state.qa_chain.invoke({"question": question})
            answer = result.get("answer", "").strip()
            if not answer:
                answer = "I don't know from the uploaded documents."

            source_documents = result.get("source_documents", [])
            sources = extract_sources(source_documents)

            with st.chat_message("assistant"):
                st.markdown(answer)
                if sources:
                    st.caption("Sources: " + " · ".join(sources))

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as exc:  # pragma: no cover - surfaced in the UI
            logger.exception("Could not answer the document question.")
            error_message = (
                "I could not answer this question with the current document index. "
                "Please verify the model is available and your PDF documents are valid."
            )
            with st.chat_message("assistant"):
                st.error(error_message)
                st.caption(f"Technical detail: {type(exc).__name__}: {exc}")
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})

st.markdown("""
---
### How the app works
1. PDFs are loaded and chunked with a recursive text splitter.
2. A local sentence-transformers model converts chunks into embeddings.
3. Chroma stores and retrieves the most relevant chunks.
4. The current question is answered using retrieved context and recent chat history.
5. Follow-up questions can reference earlier answers without losing grounding.
""")
