from app.rag.universal_loader import load_file
from app.rag.vector_store import create_vector_store
from app.rag.rag_chain import ask_rag

load_document = load_file
load_pdf = load_file

__all__ = ["load_file", "load_document", "load_pdf", "create_vector_store", "ask_rag"]
