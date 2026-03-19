import os

def load_pdf(file_path):
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(file_path)
    return loader.load()

def load_documents_from_folder(folder_path):
    """
    Load all PDF files from a folder.
    """
    docs = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)
            docs.extend(load_pdf(path))

    return docs