import streamlit as st
import os
import tempfile
import joblib
import base64
import pandas as pd
import numpy as np
from app.plagiarism_detector import get_similarity
from app.pdf_extractor import extract_text_from_pdf, store_text_in_vector_db, query_groq
from app.query_engine import query_pdf
from app.document_classifier import render_document_classification

# Page configuration
st.set_page_config(
    page_title="Document Intelligence System", 
    page_icon="📄", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #5e17eb, #10abff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.8rem;
        color: #4d4dff;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .info-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
    .stButton>button {
        background-color: #4d4dff;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3a3ad1;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

# Set theme based on sidebar selection
theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: white;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #2d2d2d;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function for download buttons
def get_download_link(text, filename, link_text):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# App header
st.markdown("<div class='main-header'>📄 Document Intelligence System</div>", unsafe_allow_html=True)
st.write("Welcome to the advanced document analysis platform powered by AI 🚀")

# Sidebar navigation
st.sidebar.markdown("### 🧭 Navigation")
option = st.sidebar.radio("Select a feature:",
                         ["Plagiarism Detection",
                          "PDF Text Extraction",
                          "Document Summarization",
                          "Document Classification",
                          "Voice Interaction"])

# Plagiarism Detection
if option == "Plagiarism Detection":
    st.markdown("<div class='subheader'>🔍 Plagiarism Checker</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Document 1")
        text1 = st.text_area("Enter or paste text:", height=250, key="text1")
        upload_option1 = st.checkbox("Upload a file instead", key="upload1")
        if upload_option1:
            uploaded_file1 = st.file_uploader("Upload document 1", type=["txt", "pdf"], key="file1")
            if uploaded_file1 is not None:
                if uploaded_file1.name.endswith('.pdf'):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                        temp_pdf.write(uploaded_file1.getbuffer())
                        save_path = temp_pdf.name
                    text1 = extract_text_from_pdf(save_path)
                    os.remove(save_path)
                else:
                    text1 = uploaded_file1.getvalue().decode("utf-8")
                st.success("Document 1 uploaded successfully!")
    
    with col2:
        st.markdown("### Document 2")
        text2 = st.text_area("Enter or paste text:", height=250, key="text2")
        upload_option2 = st.checkbox("Upload a file instead", key="upload2")
        if upload_option2:
            uploaded_file2 = st.file_uploader("Upload document 2", type=["txt", "pdf"], key="file2")
            if uploaded_file2 is not None:
                if uploaded_file2.name.endswith('.pdf'):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                        temp_pdf.write(uploaded_file2.getbuffer())
                        save_path = temp_pdf.name
                    text2 = extract_text_from_pdf(save_path)
                    os.remove(save_path)
                else:
                    text2 = uploaded_file2.getvalue().decode("utf-8")
                st.success("Document 2 uploaded successfully!")
    
    check_button = st.button("Check Plagiarism", key="check_plag")
    
    if check_button:
        if text1 and text2:
            with st.spinner("Analyzing documents for similarities..."):
                similarity = get_similarity(text1, text2)
                
                # Create visualization for similarity score
                st.markdown(f"### Similarity Results")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Create progress bar visualization
                    st.progress(similarity/100)
                    
                with col2:
                    st.markdown(f"<div style='font-size: 2rem; font-weight: bold; text-align: center;'>{similarity:.2f}%</div>", 
                                unsafe_allow_html=True)
                
                # Interpretation of similarity score
                if similarity < 20:
                    st.success("✅ Low similarity detected. Documents appear to be original.")
                elif similarity < 40:
                    st.info("ℹ️ Moderate similarity detected. Some common phrases or ideas may be present.")
                elif similarity < 60:
                    st.warning("⚠️ Significant similarity detected. Documents share substantial content.")
                else:
                    st.error("🚨 High similarity detected! Documents may be plagiarized.")
        else:
            st.warning("Please provide both texts to compare.")

# PDF Text Extraction
elif option == "PDF Text Extraction":
    st.markdown("<div class='subheader'>📄 PDF Analysis & Q&A</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Processing your PDF... This may take a moment"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_file.getbuffer())
                save_path = temp_pdf.name
            
            extracted_text = extract_text_from_pdf(save_path)
            
            tab1, tab2 = st.tabs(["Content", "Q&A"])
            
            with tab1:
                st.markdown("### Extracted Text")
                st.text_area("Document Content:", extracted_text, height=300, key="extracted_text_view")
                
                st.markdown(get_download_link(extracted_text, 
                                            f"{uploaded_file.name.split('.')[0]}_extracted.txt", 
                                            "📥 Download Extracted Text"), 
                            unsafe_allow_html=True)
                
                if st.button("Store in Vector DB", key="store_db"):
                    with st.spinner("Indexing document in database..."):
                        store_text_in_vector_db(extracted_text)
                        st.success("✅ Text stored successfully in vector database!")
            
            with tab2:
                st.markdown("### Ask Questions About Your Document")
                question = st.text_input("Enter your question:", key="pdf_question")
                
                if st.button("Get Answer", key="get_answer"):
                    if question:
                        with st.spinner("Generating answer... This may take a moment"):
                            answer = query_groq(extracted_text, question)
                            st.markdown("### 💡 AI Response:")
                            st.markdown(f"<div class='info-box'>{answer}</div>", unsafe_allow_html=True)
                            
                            # Add follow-up question suggestions
                            st.markdown("### Ask a follow-up question:")
                            follow_up_questions = [
                                "Can you summarize the main points?",
                                "What are the key findings?",
                                "How does this relate to industry standards?",
                                "What limitations are mentioned?"
                            ]
                            
                            cols = st.columns(2)
                            for i, q in enumerate(follow_up_questions):
                                if cols[i % 2].button(q, key=f"follow_up_{i}"):
                                    with st.spinner("Generating answer..."):
                                        follow_up_answer = query_groq(extracted_text, q)
                                        st.markdown("### 💡 AI Response:")
                                        st.markdown(f"<div class='info-box'>{follow_up_answer}</div>", 
                                                    unsafe_allow_html=True)
            
            os.remove(save_path)  # Clean up

# Document Summarization
elif option == "Document Summarization":
    st.markdown("<div class='subheader'>📝 Document Summarization</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a Document", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        text_to_summarize = ""
        
        if uploaded_file.name.endswith('.pdf'):
            with st.spinner("Extracting text from PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf.write(uploaded_file.getbuffer())
                    save_path = temp_pdf.name
                text_to_summarize = extract_text_from_pdf(save_path)
                os.remove(save_path)
        else:
            text_to_summarize = uploaded_file.getvalue().decode("utf-8")
        
        st.text_area("Document Content:", text_to_summarize, height=200)
        
        col1, col2 = st.columns(2)
        with col1:
            summary_type = st.radio("Summary Type:", ["Brief", "Detailed", "Bullet Points"])
        
        with col2:
            audience = st.selectbox("Target Audience:", ["General", "Academic", "Technical", "Business"])
        
        if st.button("Generate Summary", key="gen_summary"):
            with st.spinner("Generating summary... This may take a moment"):
                summary_prompt = f"Summarize the following text in a {summary_type.lower()} format for a {audience.lower()} audience:\n\n{text_to_summarize}"
                summary = query_groq(text_to_summarize, summary_prompt)
                
                st.markdown("### 📋 Summary")
                st.markdown(f"<div class='info-box'>{summary}</div>", unsafe_allow_html=True)
                
                st.markdown(get_download_link(summary, 
                                           f"{uploaded_file.name.split('.')[0]}_summary.txt", 
                                           "📥 Download Summary"), 
                           unsafe_allow_html=True)

# Document Classification - Now using the imported module
elif option == "Document Classification":
    render_document_classification()

# Voice Interaction (future feature)
elif option == "Voice Interaction":
    st.markdown("<div class='subheader'>🎤 Voice Interaction</div>", unsafe_allow_html=True)
    
    st.info("To enable voice interaction, install the following packages and uncomment the voice interaction code in app/pdf_extractor.py:")
    
    st.code("""
    pip install SpeechRecognition pydub gtts
    
    # Then uncomment the voice interaction code in app/pdf_extractor.py
    """)
    
    st.markdown("""
    ### Voice Interaction Features (Coming Soon)
    - 🗣️ Ask questions by speaking
    - 🔊 Listen to AI responses
    - 📝 Voice transcription
    - 🌐 Multi-language support
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed using Streamlit & AI Models 🚀")
st.sidebar.info("Developed by: @Ashwin Mehta🎶")

# Add helpful resources
with st.sidebar.expander("📚 Help & Resources"):
    st.markdown("""
    - [User Guide](https://example.com/guide)
    - [API Documentation](https://example.com/api)
    - [Report a Bug](https://example.com/bugs)
    - [Suggest a Feature](https://example.com/features)
    """)

# System status
with st.sidebar.expander("🔧 System Status"):
    st.markdown("✅ Plagiarism Detection: Online")
    st.markdown("✅ PDF Text Extraction: Online")
    st.markdown("✅ Vector Database: Online")
    st.markdown("✅ Groq API Connection: Online")
    st.markdown("✅ Document Classification: Online")