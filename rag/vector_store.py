import os
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embeddings):

    if os.path.exists("vector_db"):

        vector_db = FAISS.load_local(
            "vector_db",
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:

        vector_db = FAISS.from_documents(chunks, embeddings)
        vector_db.save_local("vector_db")

    return vector_db