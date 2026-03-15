import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.embeddings import get_embedding_model
from rag.retriever import load_vector_store, get_retriever
from rag.qa_chain import create_qa_chain


# Load embedding model
embeddings = get_embedding_model()

# Load vector database
vector_db = load_vector_store(embeddings)

# Create retriever
retriever = get_retriever(vector_db)

# Create QA system
qa_chain = create_qa_chain(retriever)


print("RAG system ready! Ask questions.\n")

while True:

    query = input("Question: ")

    if query.lower() == "exit":
        break

    answer = qa_chain.invoke(query)

    print("\nAnswer:", answer)
    print("\n")