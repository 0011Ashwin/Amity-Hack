import os
import re
import joblib
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
from app.pdf_extractor import extract_text_from_pdf

# Function to clean text for classification
def clean_text(text):
    """Clean text for classification"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    return text

# Attempt to load DOCX document processing
try:
    import docx
    def extract_text_from_docx(docx_file):
        """Extract text from DOCX files"""
        doc = docx.Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs])
except ImportError:
    def extract_text_from_docx(docx_file):
        """Fallback when python-docx not available"""
        return "DOCX extraction requires python-docx package. Please install it with 'pip install python-docx'."

# Add extract_features function
def extract_features(text):
    """Extract features from document text for classification"""
    if not isinstance(text, str) or not text:
        return {"word_count": 0, "avg_word_length": 0, "sentence_count": 0}
    
    # Clean the text for better feature extraction
    cleaned_text = clean_text(text)
    
    # Word count
    words = cleaned_text.split()
    word_count = len(words)
    
    # Average word length
    if word_count > 0:
        avg_word_length = sum(len(word) for word in words) / word_count
    else:
        avg_word_length = 0
    
    # Sentence count (rough estimate)
    sentence_count = len(re.split(r'[.!?]', text))
    
    return {
        "word_count": word_count,
        "avg_word_length": avg_word_length,
        "sentence_count": sentence_count
    }

def classify_document(uploaded_file):
    """Get document text"""
    # Extract text from document
    doc_text = ""
    try:
        if uploaded_file.name.endswith('.pdf'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                save_path = temp_file.name
            doc_text = extract_text_from_pdf(save_path)
            os.remove(save_path)
        elif uploaded_file.name.endswith('.docx'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                save_path = temp_file.name
            doc_text = extract_text_from_docx(save_path)
            os.remove(save_path)
        else:
            # Only decode as UTF-8 for text files
            try:
                doc_text = uploaded_file.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                doc_text = "Cannot decode file. It may be a binary file type that we don't support."
        
        # Clean text for better classification
        doc_text_cleaned = clean_text(doc_text)
        
        return doc_text, doc_text_cleaned
    except Exception as e:
        return f"Error processing document: {str(e)}", ""

# Add document_type_classifier function
def document_type_classifier(features):
    """Classify document type based on extracted features"""
    # Simple rule-based classification (placeholder for ML model)
    word_count = features.get("word_count", 0)
    avg_word_length = features.get("avg_word_length", 0)
    
    if word_count > 1000 and avg_word_length > 6:
        return "Academic Paper", 85.0
    elif word_count > 500 and avg_word_length > 5:
        return "Business Report", 75.0
    elif word_count > 200:
        return "Article", 70.0
    else:
        return "General Document", 60.0

def render_document_classification():
    """Render the document classification interface"""
    st.markdown("<div class='subheader'>📊 Document Classification</div>", unsafe_allow_html=True)
    
    st.markdown("""
    Upload a document to classify its type and content category.
    Supported formats: PDF, TXT
    """)
    
    uploaded_file = st.file_uploader("Upload a Document", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing document..."):
                doc_text, doc_text_cleaned = classify_document(uploaded_file)
            
            if doc_text and isinstance(doc_text, str) and not doc_text.startswith("Error"):
                # Extract features for classification
                features = extract_features(doc_text)
                
                # Perform classification
                doc_type, confidence_score = document_type_classifier(features)
                
                # Display results
                st.success("Document processed successfully!")
                st.markdown("### Classification Results")
                
                st.markdown(f"""
                <div class="card light-card">
                    <h4>Document Type: {doc_type}</h4>
                    <p>Classification confidence: {confidence_score:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.error(f"Could not process document content: {doc_text}")
                
        except Exception as e:
            st.error(f"Error in classification: {str(e)}")
            st.info("Please try uploading a different document or contact support.")
