import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
import model  # Ensure this import is present

def process_pdf_for_code_generator(uploaded_file):
    """
    Processes the uploaded PDF file for storage

    Args:
        uploaded_file -> file for RAG ingestion pipeline

    Returns:
        splits -> Str
    """
    temp_file_path = f"temp_{uploaded_file.name}"
    try:
        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load Documents
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        # Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        return splits
    finally:
        # Ensure the temp file is deleted
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def store_pdf_in_chroma(uploaded_file, vectorstore):
    """
    Stores the content of the uploaded PDF file in a local ChromaDB.

    Args:
        uploaded_file -> file for RAG ingestion pipeline
        vectorstore ->  Instance of vector store        

    Returns:
        vectorstore -> Instance of vector store        
    """
    try:
        splits = process_pdf_for_code_generator(uploaded_file)
        # Embed and store in local ChromaDB
        vectorstore.add_documents(splits)
    except Exception as e:
        st.error(f"An error occurred while processing the uploaded file: {e}")
def initialize_chroma(persist_directory="./chroma_db"):
    """
    Initializes and returns a Chroma vector store.

    Args:
        persist_directory - Directory to store ChromaDB.
    
    Returns:
        vectorstore - Initialized Chroma vector store.
    """
    hf_embeddings = model.create_hugging_face_embedding_model()
    vectorstore = Chroma(embedding_function=hf_embeddings, persist_directory=persist_directory)
    vectorstore._persist_directory
    return vectorstore
def retrieve_from_chroma(query, vectorstore):
    """
    Retrieves the most relevant documents from the Chroma vector store
    based on the user's query.

    Args:
        query -> The query string for searching the vector store.
        vectorstore -> The Chroma vector store instance for document retrieval.

    Returns:
        documents - The most relevant documents retrieved from Chroma.
    """
    # Convert the query dictionary to a string format
    query_string = " ".join(f"{key}: {value}" for key, value in query.items())
    
    retriever = vectorstore.as_retriever()
    documents = retriever.get_relevant_documents(query_string)
    return