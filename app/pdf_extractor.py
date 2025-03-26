import pdfplumber
import os
from dotenv import load_dotenv
import groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("❌ Groq API key is missing! Set GROQ_API_KEY in .env file.")

# Initialize Groq Client
groq_client = groq.Client(api_key=groq_api_key)

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
    try:
        # Check if text is too long
        if len(text) > 6000:
            truncated_text = text[:6000]
            st.warning("Text was truncated to 6,000 characters due to API limits.")
        else:
            truncated_text = text
            
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Use this model for best results
            messages=[
                {"role": "system", "content": "You are an AI assistant that helps users extract information from documents. Provide concise, accurate answers based solely on the document content provided."},
                {"role": "user", "content": f"Context: {truncated_text}\n\nQuestion: {question}"}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content if response.choices else "No response available."
    except Exception as e:
        # If the model fails, try with a smaller model
        try:
            st.warning(f"Error with primary model: {str(e)}. Trying alternative model...")
            response = groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",  # Fallback model
                messages=[
                    {"role": "system", "content": "You are an AI assistant that helps users extract information from documents."},
                    {"role": "user", "content": f"Context: {text[:4000]}\n\nQuestion: {question}"}
                ],
                max_tokens=800
            )
            return response.choices[0].message.content if response.choices else "No response available."
        except Exception as fallback_error:
            return f"Error querying AI: {str(fallback_error)}"

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
