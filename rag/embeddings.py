from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_model():
    """
    Load embedding model used to convert text into vectors.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings