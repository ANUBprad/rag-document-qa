from langchain_community.vectorstores import FAISS


def load_vector_store(embeddings):
    """
    Load existing FAISS vector database
    """

    vector_db = FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_db


def get_retriever(vector_db):

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever