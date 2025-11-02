# app.py
import streamlit as st
from ingest import load_file, split_docs, embed_and_store
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import google.generativeai as genai   
import os
from datetime import datetime

# Load environment variables
load_dotenv()

# Get Gemini API key
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    st.error("GEMINI_API_KEY not found in .env file.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=gemini_key)

# Page title
st.title("RAG Demo — Upload a document & ask questions")

# Create folders
UPLOAD_DIR = "uploads"
CHROMA_BASE_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_BASE_DIR, exist_ok=True)

# File uploader
uploaded_file = st.file_uploader(" Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])

# Variable to hold current Chroma path
persist_dir = None

if uploaded_file is not None:
    # Save uploaded file locally
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded successfully: {uploaded_file.name}")

    # Create a new subfolder for this upld
    folder_name = uploaded_file.name.split('.')[0] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    persist_dir = os.path.join(CHROMA_BASE_DIR, folder_name)
    os.makedirs(persist_dir, exist_ok=True)

    # Process and store embedding
    docs = load_file(file_path)
    chunks = split_docs(docs)
    db = embed_and_store(chunks, persist_dir=persist_dir)
    st.success(f"Document processed and stored in {persist_dir}")

# Query input
query = st.text_input("Ask a question about the uploaded document")

if query:
    all_dirs = [os.path.join(CHROMA_BASE_DIR, d) for d in os.listdir(CHROMA_BASE_DIR)]
    latest_dir = max(all_dirs, key=os.path.getmtime) if all_dirs else None

    if not latest_dir:
        st.error("No vector database found. Please upload and ingest a document first.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=latest_dir, embedding_function=embeddings)
    results = vectordb.similarity_search(query, k=4)

    context_text = "\n\n---\n\n".join([d.page_content for d in results])

    prompt = f"""
You are a helpful assistant that answers questions using ONLY the provided context.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context_text}

Question: {query}

Answer concisely, mention the source chunk numbers, and give a short confidence score (low/medium/high).
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        st.error(f"---- Gemini API Error: ----{e}")
        st.stop()

    # Display result
    st.subheader("Answer:")
    st.write(answer)

    st.subheader("Retrieved Chunks Used:")
    for i, d in enumerate(results):
        st.markdown(f"**Chunk {i+1} — {d.metadata.get('source', '-')}:**")
        st.write(d.page_content[:300] + "...")

st.markdown("---")
st.markdown("<center>✨ Developed by Priyanshu Panda ✨</center>", unsafe_allow_html=True)
