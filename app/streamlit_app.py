import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from rag.document_loader import load_pdf
from rag.chunker import split_documents
from rag.embeddings import get_embedding_model
from rag.vector_store import create_vector_store
from rag.retriever import get_retriever
from rag.qa_chain import create_qa_chain


# Page config
st.set_page_config(page_title="RAG Document QA", layout="wide")

# Title
st.title("📄 AI Document Assistant")
st.write("Upload a PDF and ask questions based on its content.")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Session state 
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
    st.session_state.retriever = None

if uploaded_file:

    # Save uploaded file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Build pipeline only once
    if st.session_state.qa_chain is None:

        with st.spinner("Processing document... ⏳"):

            docs = load_pdf("temp.pdf")
            chunks = split_documents(docs)

            embeddings = get_embedding_model()
            vector_db = create_vector_store(chunks, embeddings)

            retriever = get_retriever(vector_db)
            qa_chain = create_qa_chain(retriever)

            # Save in session
            st.session_state.qa_chain = qa_chain
            st.session_state.retriever = retriever

        st.success("✅ Document processed successfully!")

    # Input question
    question = st.text_input("Ask a question")

    if question:

        retriever = st.session_state.retriever
        qa_chain = st.session_state.qa_chain

        # Retrieve relevant docs
        docs = retriever.get_relevant_documents(question)

        # Get answer
        answer, docs = qa_chain(question, mode, doc_type)

        # Display answer
        st.subheader("🧠 Answer")
        st.write(answer)

        # Display sources
        st.subheader("📚 Sources")

        for i, doc in enumerate(docs):
            st.markdown(f"**Source {i+1}:**")
            st.info(doc.page_content[:300])