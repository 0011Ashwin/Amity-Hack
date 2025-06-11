import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("❌ Groq API key is missing! Set GROQ_API_KEY in .env file.")

# Function to get the Groq client
def get_groq_client():
    """Initialize and return a Groq client instance safely."""
    try:
        # First try importing the module
        import groq
        
        # Simple client initialization
        client = groq.Client(api_key=groq_api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
        return None

# Function to query Groq
def query_groq_api(text, question):
    """Send extracted text to Groq API and get a response."""
    client = get_groq_client()
    
    if not client:
        return "Could not initialize Groq client."
    
    try:
        # Check if text is too long
        if len(text) > 6000:
            truncated_text = text[:6000]
            st.warning("Text was truncated to 6,000 characters due to API limits.")
        else:
            truncated_text = text
            
        response = client.chat.completions.create(
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
            response = client.chat.completions.create(
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