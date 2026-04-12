from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pandas as pd
from langchain_core.documents import Document

_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
}


def _build_document(page_content: str, file_path: str, loader_name: str) -> list[Document]:
    cleaned_content = page_content.strip()
    if not cleaned_content:
        return []

    source_path = os.path.abspath(file_path)
    file_name = os.path.basename(source_path)
    return [
        Document(
            page_content=cleaned_content,
            metadata={
                "file_name": file_name,
                "source": source_path,
                "loader": loader_name,
            },
        )
    ]


def _is_probably_text(text: str) -> bool:
    stripped_text = text.strip()
    if not stripped_text:
        return False

    printable_count = sum(1 for char in stripped_text if char.isprintable() or char.isspace())
    return printable_count / max(len(stripped_text), 1) >= 0.85


def try_text_read(file_path: str) -> list[Document]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file_handle:
            content = file_handle.read()
    except OSError as exc:
        print(f"[RAG] Plain text read failed: {exc}")
        return []

    if not _is_probably_text(content):
        print("[RAG] Plain text read did not produce usable text.")
        return []

    documents = _build_document(content, file_path, "plain_text")
    if documents:
        print(f"[RAG] Loaded file as plain text: {file_path}")
    return documents


def try_unstructured_loader(file_path: str) -> list[Document]:
    try:
        from langchain_unstructured import UnstructuredLoader
        from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
    except Exception as exc:
        print(f"[RAG] Unstructured loader imports unavailable: {exc}")
        return []

    loader_attempts: list[tuple[str, Callable[[], list[Document]]]] = []

    extension = Path(file_path).suffix.lower()
    if extension in _IMAGE_EXTENSIONS:
        print("[RAG] Skipping unstructured loader for image file.")
        return []

    # Skip unstructured for spreadsheet formats (pandas is better for these)
    if extension in {".xlsx", ".xls", ".xlsm", ".csv", ".tsv"}:
        print(f"[RAG] Skipping unstructured loader for spreadsheet format {extension}.")
        return []

    if extension == ".pdf":
        loader_attempts.append(("pdf", lambda: PyPDFLoader(file_path).load()))
    elif extension == ".docx":
        loader_attempts.append(("docx", lambda: Docx2txtLoader(file_path).load()))
    elif extension in {".txt", ".md", ".rst", ".py", ".json", ".yaml", ".yml"}:
        loader_attempts.append(("text", lambda: TextLoader(file_path, encoding="utf-8").load()))

    if extension != ".pdf":
        try:
            loader_attempts.append(("unstructured", lambda: UnstructuredLoader(file_path=file_path).load()))
        except Exception:
            pass

    for loader_name, loader_callable in loader_attempts:
        try:
            documents = loader_callable()
            if documents:
                normalized_documents = []
                for document in documents:
                    content = getattr(document, "page_content", "")
                    normalized_documents.extend(_build_document(content, file_path, loader_name))

                if normalized_documents:
                    print(f"[RAG] Loaded file with {loader_name} loader: {file_path}")
                    return normalized_documents
        except Exception as exc:
            print(f"[RAG] {loader_name} loader failed: {exc}")

    return []


def try_pandas(file_path: str) -> list[Document]:
    try:
        dataframes = []
        sheet_names = []
        extension = Path(file_path).suffix.lower()

        if extension in {".xlsx", ".xls", ".xlsm"}:
            # Load all sheets from Excel file
            try:
                excel_file = pd.ExcelFile(file_path)
                sheet_names = excel_file.sheet_names
                for sheet_name in sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    if not df.empty:
                        dataframes.append((sheet_name, df))
            except Exception:
                dataframe = pd.read_excel(file_path)
                if not dataframe.empty:
                    dataframes.append(("Sheet1", dataframe))
        elif extension == ".csv":
            df = pd.read_csv(file_path)
            if not df.empty:
                dataframes.append(("Data", df))
        elif extension == ".tsv":
            df = pd.read_csv(file_path, sep="\t")
            if not df.empty:
                dataframes.append(("Data", df))
        else:
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    dataframes.append(("Data", df))
            except Exception:
                try:
                    df = pd.read_excel(file_path)
                    if not df.empty:
                        dataframes.append(("Data", df))
                except Exception:
                    pass

        if not dataframes:
            print("[RAG] Pandas did not produce usable tabular content.")
            return []

        # Combine all sheets into a single content string with sheet headers
        combined_content_parts = []
        for sheet_name, dataframe in dataframes:
            # Format numeric columns for better readability and semantic search
            formatted_df = dataframe.copy()
            for col in formatted_df.columns:
                if formatted_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Convert to string with appropriate precision, avoiding scientific notation
                    formatted_df[col] = formatted_df[col].apply(
                        lambda x: f"{x:.6g}" if pd.notna(x) else "N/A"  # 6 significant figures, no sci notation
                    )

            sheet_content = formatted_df.to_string(index=False)
            combined_content_parts.append(f"=== Sheet: {sheet_name} ===\n{sheet_content}")

        combined_content = "\n\n".join(combined_content_parts)
        documents = _build_document(combined_content, file_path, "pandas")
        if documents:
            if len(dataframes) > 1:
                print(f"[RAG] Loaded tabular file with {len(dataframes)} sheets via pandas: {file_path}")
            else:
                print(f"[RAG] Loaded tabular file with pandas: {file_path}")
        return documents
    except Exception as exc:
        print(f"[RAG] Pandas parsing failed: {exc}")
        return []


def try_ocr(file_path: str) -> list[Document]:
    try:
        from PIL import Image, ImageEnhance, ImageOps
        import pytesseract
    except Exception as exc:
        print(f"[RAG] OCR dependencies unavailable: {exc}")
        return []

    try:
        image = Image.open(file_path)
    except Exception as exc:
        print(f"[RAG] OCR failed: {exc}")
        return []

    # Try multiple OCR-friendly variants before giving up.
    variants = []

    base = image.convert("RGB")
    variants.append(base)

    gray = ImageOps.grayscale(base)
    variants.append(gray)

    enhanced = ImageEnhance.Contrast(gray).enhance(2.0)
    variants.append(enhanced)

    binary = enhanced.point(lambda px: 255 if px > 160 else 0)
    variants.append(binary)

    upscaled = binary.resize((max(1, binary.width * 2), max(1, binary.height * 2)))
    variants.append(upscaled)

    extracted_text = ""
    for variant in variants:
        try:
            candidate = pytesseract.image_to_string(variant, config="--oem 3 --psm 6").strip()
            if candidate:
                extracted_text = candidate
                break
        except Exception:
            continue

    if not extracted_text:
        # Last attempt using sparse text mode for posters/charts.
        try:
            extracted_text = pytesseract.image_to_string(base, config="--oem 3 --psm 11").strip()
        except Exception as exc:
            print(f"[RAG] OCR failed: {exc}")
            return []

    if not extracted_text:
        print("[RAG] OCR ran successfully but no readable text was found in the image.")
        return []

    documents = _build_document(extracted_text, file_path, "ocr")
    if documents:
        print(f"[RAG] Loaded image via OCR: {file_path}")
    return documents


def load_file(file_path: str) -> list[Document]:
    normalized_path = os.path.abspath(os.path.expanduser(str(file_path).strip()))

    if not normalized_path:
        raise FileNotFoundError("File not found")

    if not os.path.exists(normalized_path):
        raise FileNotFoundError(f"File not found: {normalized_path}")

    print(f"[RAG] Starting ingestion for: {normalized_path}")

    extension = Path(normalized_path).suffix.lower()
    
    # Skip plain text reading for binary formats
    _BINARY_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".xlsm", ".pptx", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
    
    if extension in _IMAGE_EXTENSIONS:
        loaders = (
            try_text_read,
            try_ocr,
            try_unstructured_loader,
            try_pandas,
        )
    elif extension in _BINARY_EXTENSIONS:
        # Skip text read for binary formats; use unstructured loaders first
        loaders = (
            try_unstructured_loader,
            try_pandas,
            try_ocr,
            try_text_read,
        )
    else:
        loaders = (
            try_text_read,
            try_unstructured_loader,
            try_pandas,
            try_ocr,
        )

    for loader in loaders:
        try:
            documents = loader(normalized_path)
            if documents:
                return documents
        except Exception as exc:
            print(f"[RAG] Loader {loader.__name__} raised an error: {exc}")

    if extension in _IMAGE_EXTENSIONS:
        raise ValueError("No readable text found in image")

    raise ValueError("Unsupported or unreadable file format")
