from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def _get_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def create_vector_store(docs: list[Document]) -> FAISS | None:
    if not docs:
        return None

    chunks = _TEXT_SPLITTER.split_documents(docs)
    if not chunks:
        return None

    try:
        embeddings = _get_embeddings()
        return FAISS.from_documents(chunks, embeddings)
    except Exception:
        return None
