import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.document_loader import load_documents_from_folder
from rag.chunker import split_documents
from rag.embeddings import get_embedding_model
from rag.vector_store import create_vector_store


folder_path = "data/documents"

# Load documents
documents = load_documents_from_folder(folder_path)
print("Documents Loaded:", len(documents))

# Split documents
chunks = split_documents(documents)
print("Chunks Created:", len(chunks))

# Load embedding model
embeddings = get_embedding_model()

# Create vector database
vector_db = create_vector_store(chunks, embeddings)

print("Vector database created successfully!")