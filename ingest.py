# ingest.py
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def load_file(path):
    if path.endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return [{"page_content": f.read(), "metadata": {}}]
    return loader.load()

def split_docs(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def embed_and_store(docs, persist_dir="chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb


# docs = load_file("mydoc.pdf")
# chunks = split_docs(docs)
# db = embed_and_store(chunks)
