from dotenv import load_dotenv
import streamlit as st
from chain import generate_code, generate_code_with_rag
import vectordb

load_dotenv()

def code_generator():
    """
    Code Generator Bot with RAG
    """
    st.sidebar.title("Menu")
    section = st.sidebar.radio(
        "Choose a section:",
        ("Code Generator RAG", "RAG File Ingestion")
    )

    # db initialization
    vectordatabase = vectordb.initialize_chroma()

    # Condition for code generation page
    if section == "Code Generator RAG":
        st.title("Code Generator Bot")

        with st.form("Code_Generator"):
            language = st.text_input("Enter the Programming Language")
            problem_statement = st.text_input("Enter the Problem Statement")
            submitted = st.form_submit_button("Submit", type="primary")
            toggle_state = st.checkbox("Check me to enable RAG")

            if submitted:
                if toggle_state:
                    response = generate_code_with_rag(language, problem_statement, vectordatabase)
                else:
                    response = generate_code(language, problem_statement)

                st.info(response)

    # Condition for RAG File Ingestion
    elif section == "RAG File Ingestion":
        st.title("RAG File Ingestion")

        uploaded_file = st.file_uploader("Upload a file:", type=["txt", "csv", "docx", "pdf"])

        if uploaded_file is not None:
            print("start")
            vectordb.store_pdf_in_chroma(uploaded_file, vectordatabase)
            print("end")
            st.success(f"File '{uploaded_file.name}' uploaded and file embedding stored in vectordb successfully!")

if __name__ == "__main__":
    code_generator()
