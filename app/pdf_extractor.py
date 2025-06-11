import pdfplumber
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.groq_utils import query_groq_api

# Load environment variables
load_dotenv()

# Initialize Vector Database
vector_db = None

def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
    except Exception as e:
        return f"Error extracting text: {str(e)}"
    
    return text.strip()

def store_text_in_vector_db(text):
    """Split text into chunks and store in a vector database."""
    global vector_db
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_chunks = text_splitter.split_text(text)

        vector_db = FAISS.from_texts(text_chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        return "Text stored in vector DB successfully."
    except Exception as e:
        return f"Error storing text in vector DB: {str(e)}"

def retrieve_relevant_text(query):
    """Retrieve relevant text chunks from the vector database."""
    if vector_db is None:
        return None
    
    try:
        retrieved_chunks = vector_db.similarity_search(query, k=5)
        return " ".join([chunk.page_content for chunk in retrieved_chunks])
    except Exception as e:
        st.error(f"Error retrieving text: {str(e)}")
        return None

def query_groq(text, question):
    """Send extracted text to Groq API and get a response."""
    return query_groq_api(text, question)

def hybrid_query_pipeline(pdf_path, question):
    """Full pipeline: extract text, store in DB, retrieve, and query LLM."""
    extracted_text = extract_text_from_pdf(pdf_path)
    store_text_in_vector_db(extracted_text)  # Store in vector DB

    relevant_text = retrieve_relevant_text(question)  # Get relevant context
    if relevant_text:  # If relevant text is found in the database
        return query_groq(relevant_text, question)
    else:  # Otherwise, send the full text to LLM
        return query_groq(extracted_text, question)

# Example usage
if __name__ == "__main__":
    pdf_path = "data/pdf-1.pdf"
    question = "Summarize the document."
    
    print("\nExtracting and Processing PDF...")
    answer = hybrid_query_pipeline(pdf_path, question)
    
    print("\n💡 AI Response:")
    print(answer)
