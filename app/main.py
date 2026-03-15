import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.document_loader import load_documents_from_folder
from rag.chunker import split_documents

folder_path = "data/documents"

documents = load_documents_from_folder(folder_path)

print("Documents Loaded:", len(documents))

chunks = split_documents(documents)

print("Chunks Created:", len(chunks))