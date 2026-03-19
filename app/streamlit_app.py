import streamlit as st
import tempfile

from rag.document_loader import load_pdf
from rag.chunker import split_documents
from rag.embeddings import get_embedding_model
from rag.vector_store import create_vector_store
from rag.retriever import get_retriever
from rag.qa_chain import create_qa_chain
from rag.doc_classifier import classify_document

st.set_page_config(page_title="RAG Document QA")

st.title("📄 AI Document Assistant")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    # 🔥 Save file uniquely every time
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name

    # 🔥 Fresh pipeline (NO CACHE)
    docs = load_pdf(temp_path)
    chunks = split_documents(docs)

    embeddings = get_embedding_model()
    vector_db = create_vector_store(chunks, embeddings)
    retriever = get_retriever(vector_db)

    qa_chain = create_qa_chain(retriever)

    doc_type = classify_document(docs)

    st.success(f"📂 Document Type: {doc_type}")

    # Debug (optional)
    # st.write("Chunks:", len(chunks))
    # st.write("Sample:", chunks[0].page_content[:200])

    question = st.text_input("Ask a question")

    if question:

        answer, suggestions, intent = qa_chain(question, doc_type)

        st.write("### 🧠 Intent:", intent)

        st.write("### Answer")
        st.write(answer)

        st.write("### 🔥 Suggested Questions")

        for i, q in enumerate(suggestions):
            if st.button(q, key=f"suggestion_{i}"):
                answer, suggestions, intent = qa_chain(q, doc_type)
                st.write("### Answer")
                st.write(answer)