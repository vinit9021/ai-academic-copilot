from __future__ import annotations

import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    normalized_path = str(file_path).strip()

    if not normalized_path or not os.path.isfile(normalized_path):
        return []

    try:
        loader = PyPDFLoader(normalized_path)
        documents = loader.load()
        return documents if documents else []
    except Exception:
        return []
