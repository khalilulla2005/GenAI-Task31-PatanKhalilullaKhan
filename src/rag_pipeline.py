from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
RETRIEVER_K = 4


def load_uploaded_documents(file_paths: Sequence[str | Path]) -> list[Document]:
    documents: list[Document] = []
    for path in file_paths:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Document was not found: {resolved}")

        if resolved.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(resolved))
            documents.extend(loader.load())
        elif resolved.suffix.lower() in {".txt", ".md"}:
            documents.append(Document(page_content=resolved.read_text(encoding="utf-8"), metadata={"source": str(resolved)}))
        else:
            raise ValueError(f"Unsupported document type: {resolved.suffix}")

    if not documents:
        raise ValueError("No documents were found in the uploaded files.")

    return documents


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and chunk_overlap must be between 0 and chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(list(documents))


def build_vector_store(
    documents: Sequence[Document],
    embeddings: Embeddings | None = None,
) -> tuple[Chroma, int]:
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("The uploaded documents contain no text to index.")

    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store, len(chunks)


def build_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful PDF Q&A assistant. Use only the retrieved document context to answer. "
                "Retrieved document context:\n{context}\n\n"
                "If the answer is not present in the supplied context, clearly say 'I don't know' and do not invent details.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )


def build_llm(provider: str = "ollama"):
    provider_name = provider.lower()
    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Use Ollama or add your API key.")
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            api_key=api_key,
        )

    if provider_name == "ollama":
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            temperature=0,
        )

    raise ValueError(f"Unsupported provider: {provider}")


def create_qa_chain(vector_store: Chroma, llm):
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
        k=5,
    )
    prompt = build_prompt_template()

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False,
    )


def extract_sources(source_documents: Sequence[Document]) -> list[str]:
    sources: list[str] = []
    for document in source_documents or []:
        source = str(document.metadata.get("source", "Uploaded document"))
        page = document.metadata.get("page")
        if isinstance(page, int):
            page += 1
        label = f"{Path(source).name}, p. {page}" if page is not None else Path(source).name
        if label not in sources:
            sources.append(label)
    return sources
