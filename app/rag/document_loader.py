from __future__ import annotations

import os
from typing import Callable

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

_SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".json", ".py", ".yaml", ".yml"}


def _get_loader(file_path: str) -> Callable[[], list[Document]] | None:
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return lambda: PyPDFLoader(file_path).load()

    if extension == ".docx":
        return lambda: Docx2txtLoader(file_path).load()

    if extension in _SUPPORTED_TEXT_EXTENSIONS:
        return lambda: TextLoader(file_path, encoding="utf-8").load()

    return None


def load_document(file_path: str) -> list[Document]:
    normalized_path = str(file_path).strip()

    if not normalized_path or not os.path.isfile(normalized_path):
        return []

    loader = _get_loader(normalized_path)
    if loader is None:
        return []

    try:
        documents = loader()
        return documents if documents else []
    except Exception:
        return []
