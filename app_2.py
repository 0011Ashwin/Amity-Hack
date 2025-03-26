import streamlit as st
import os
import tempfile
import base64
import pandas as pd
import numpy as np
from app.plagiarism_detector import get_similarity
from app.pdf_extractor import extract_text_from_pdf, store_text_in_vector_db, query_groq
from app.document_classifier import render_document_classification

# Page configuration and theme settings
st.set_page_config(
    page_title="AI-Powered Document Analysis System", 
    page_icon="📄", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for beautiful UI
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
    .card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
    }
    .light-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
    }
    .dark-card {
        background-color: #2d2d2d;
        border: 1px solid #444;
        color: white;
    }
    .progress-container {
        height: 30px;
        background-color: #e9ecef;
        border-radius: 15px;
        margin: 20px 0;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        border-radius: 15px;
        text-align: center;
        line-height: 30px;
        color: white;
        font-weight: bold;
    }
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    .loading-animation {
        animation: pulse 1.5s infinite;
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
    """Create a download link for text content"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}" style="display: inline-block; padding: 10px 15px; background-color: #4d4dff; color: white; text-decoration: none; border-radius: 5px; font-weight: bold;">{link_text}</a>'
    return href

# Helper function to create card UI elements
def create_card(title, content, icon=None):
    """Create a styled card component"""
    card_class = "dark-card" if theme == "Dark" else "light-card"
    icon_html = f'<div style="font-size: 2.5rem; margin-bottom: 15px;">{icon}</div>' if icon else ''
    
    st.markdown(f"""
    <div class="card {card_class}">
        {icon_html}
        <h3>{title}</h3>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)

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
    
    st.markdown("""
    This tool compares two documents and identifies similarities. 
    Upload two files or paste text directly to analyze plagiarism.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Document 1")
        text1 = st.text_area("Enter or paste text:", height=250, key="text1")
        upload_option1 = st.checkbox("Upload a file instead", key="upload1")
        if upload_option1:
            uploaded_file1 = st.file_uploader("Upload document 1", type=["txt", "pdf"], key="file1")
            if uploaded_file1 is not None:
                try:
                    if uploaded_file1.name.endswith('.pdf'):
                        with st.spinner("Extracting text from PDF..."):
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                                temp_pdf.write(uploaded_file1.getbuffer())
                                save_path = temp_pdf.name
                            text1 = extract_text_from_pdf(save_path)
                            os.remove(save_path)
                    else:
                        text1 = uploaded_file1.getvalue().decode("utf-8")
                    st.success("Document 1 uploaded successfully!")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    with col2:
        st.markdown("### Document 2")
        text2 = st.text_area("Enter or paste text:", height=250, key="text2")
        upload_option2 = st.checkbox("Upload a file instead", key="upload2")
        if upload_option2:
            uploaded_file2 = st.file_uploader("Upload document 2", type=["txt", "pdf"], key="file2")
            if uploaded_file2 is not None:
                try:
                    if uploaded_file2.name.endswith('.pdf'):
                        with st.spinner("Extracting text from PDF..."):
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                                temp_pdf.write(uploaded_file2.getbuffer())
                                save_path = temp_pdf.name
                            text2 = extract_text_from_pdf(save_path)
                            os.remove(save_path)
                    else:
                        text2 = uploaded_file2.getvalue().decode("utf-8")
                    st.success("Document 2 uploaded successfully!")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    check_button = st.button("Check Plagiarism", key="check_plag")
    
    if check_button:
        if text1 and text2:
            with st.spinner("Analyzing documents for similarities..."):
                try:
                    similarity = get_similarity(text1, text2)
                    
                    # Create visualization for similarity score
                    st.markdown(f"### Similarity Results")
                    
                    # Custom progress bar with color gradient
                    progress_color = "#4CAF50" if similarity < 30 else "#FFC107" if similarity < 60 else "#F44336"
                    st.markdown(f"""
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {similarity}%; background-color: {progress_color};">
                            {similarity:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Interpretation of similarity score
                    if similarity < 20:
                        result_icon = "✅"
                        result_title = "Low Similarity"
                        result_desc = "Documents appear to be original and distinct from each other."
                        result_color = "#d4edda"
                        text_color = "#155724"
                    elif similarity < 40:
                        result_icon = "ℹ️"
                        result_title = "Moderate Similarity"
                        result_desc = "Some common phrases or ideas may be present. Further review recommended."
                        result_color = "#d1ecf1"
                        text_color = "#0c5460"
                    elif similarity < 60:
                        result_icon = "⚠️"
                        result_title = "Significant Similarity"
                        result_desc = "Documents share substantial content. Check for proper citations or paraphrasing."
                        result_color = "#fff3cd"
                        text_color = "#856404"
                    else:
                        result_icon = "🚨"
                        result_title = "High Similarity"
                        result_desc = "Documents may be plagiarized. Immediate review required."
                        result_color = "#f8d7da"
                        text_color = "#721c24"
                    
                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: {result_color}; color: {text_color}; margin: 20px 0;">
                        <div style="display: flex; align-items: center;">
                            <div style="font-size: 2.5rem; margin-right: 20px;">{result_icon}</div>
                            <div>
                                <h3 style="margin: 0;">{result_title} Detected</h3>
                                <p style="margin-top: 5px;">{result_desc}</p>
                                <div style="font-weight: bold; font-size: 1.2rem; margin-top: 10px;">Similarity Score: {similarity:.2f}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add detailed comparison
                    st.markdown("### Detailed Comparison")
                    
                    # Show common phrases/words
                    with st.expander("View Sample Similarities"):
                        words1 = set(text1.lower().split())
                        words2 = set(text2.lower().split())
                        common_words = words1.intersection(words2)
                        
                        # Show most significant common words (exclude very common words)
                        common_words_list = list(common_words)
                        common_words_list.sort(key=lambda x: len(x), reverse=True)
                        significant_words = [w for w in common_words_list if len(w) > 4][:20]
                        
                        if significant_words:
                            st.markdown("#### Common Significant Terms:")
                            cols = st.columns(4)
                            for i, word in enumerate(significant_words):
                                cols[i % 4].markdown(f"- {word}")
                        else:
                            st.write("No significant common terms found.")
                            
                except Exception as e:
                    st.error(f"Error analyzing similarity: {str(e)}")
        else:
            st.warning("Please provide both texts to compare.")

# PDF Text Extraction
elif option == "PDF Text Extraction":
    st.markdown("<div class='subheader'>📄 PDF Analysis & Q&A</div>", unsafe_allow_html=True)
    
    st.markdown("""
    Upload a PDF file to extract text and ask questions about its content. 
    Our AI will analyze the document and provide answers based on the content.
    """)
    
    uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Processing your PDF... This may take a moment"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf.write(uploaded_file.getbuffer())
                    save_path = temp_pdf.name
                
                extracted_text = extract_text_from_pdf(save_path)
                
                # Check if extraction was successful
                if extracted_text and not extracted_text.startswith("Error"):
                    st.success(f"Successfully extracted text from {uploaded_file.name}")
                    
                    tab1, tab2 = st.tabs(["Content", "Q&A"])
                    
                    with tab1:
                        st.markdown("### Extracted Text")
                        
                        # Truncate text for display if too long
                        display_text = extracted_text
                        if len(extracted_text) > 10000:
                            display_text = extracted_text[:10000] + "... (truncated for display)"
                            st.info("Document is large. Displaying first 10,000 characters.")
                            
                        st.text_area("Document Content:", display_text, height=300, key="extracted_text_view")
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.markdown(get_download_link(extracted_text, 
                                                        f"{uploaded_file.name.split('.')[0]}_extracted.txt", 
                                                        "📥 Download Extracted Text"), 
                                    unsafe_allow_html=True)
                        
                        with col2:
                            if st.button("Store in Vector DB", key="store_db"):
                                with st.spinner("Indexing document in database..."):
                                    store_text_in_vector_db(extracted_text)
                                    st.success("✅ Text stored successfully in vector database!")
                    
                    with tab2:
                        st.markdown("### Ask Questions About Your Document")
                        st.info("Type a question about the document content, and our AI will provide an answer.")
                        
                        question = st.text_input("Enter your question:", key="pdf_question", 
                                                 placeholder="E.g., What is the main topic of this document?")
                        
                        if st.button("Get Answer", key="get_answer"):
                            if question:
                                with st.spinner("Generating answer... This may take a moment"):
                                    try:
                                        # Limit text if needed to avoid API errors
                                        if len(extracted_text) > 6000:
                                            processing_text = extracted_text[:6000]
                                            st.info("Document is very large. Using first 6,000 characters for analysis.")
                                        else:
                                            processing_text = extracted_text
                                            
                                        answer = query_groq(processing_text, question)
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
                                                    follow_up_answer = query_groq(processing_text, q)
                                                    st.markdown("### 💡 AI Response:")
                                                    st.markdown(f"<div class='info-box'>{follow_up_answer}</div>", 
                                                                unsafe_allow_html=True)
                                    except Exception as e:
                                        st.error(f"Error generating answer: {str(e)}")
                                        st.info("Try simplifying your question or using a shorter document.")
                            else:
                                st.warning("Please enter a question.")
                else:
                    st.error(f"Could not extract text from PDF: {extracted_text}")
                
                os.remove(save_path)  # Clean up
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.info("Make sure the PDF is properly formatted and not password-protected.")

# Document Summarization
elif option == "Document Summarization":
    st.markdown("<div class='subheader'>📝 Document Summarization</div>", unsafe_allow_html=True)
    
    st.markdown("""
    Upload a document to get an AI-generated summary. You can customize the summary type and target audience.
    """)
    
    uploaded_file = st.file_uploader("Upload a Document", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        try:
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
            
            # Check if extraction was successful
            if text_to_summarize and not text_to_summarize.startswith("Error"):
                # Truncate text if too long
                if len(text_to_summarize) > 10000:
                    displayed_text = text_to_summarize[:10000] + "... (truncated for display)"
                    st.warning("Document is large. Displaying first 10,000 characters.")
                else:
                    displayed_text = text_to_summarize
                    
                st.text_area("Document Content:", displayed_text, height=200)
                
                # Configure summary options
                st.markdown("### Summary Options")
                
                col1, col2 = st.columns(2)
                with col1:
                    summary_type = st.radio("Summary Type:", ["Brief", "Detailed", "Bullet Points"])
                    st.markdown(f"<small><em>{summary_type} summary provides {'a concise overview' if summary_type == 'Brief' else 'comprehensive details' if summary_type == 'Detailed' else 'key points in bullet format'}.</em></small>", unsafe_allow_html=True)
                
                with col2:
                    audience = st.selectbox("Target Audience:", ["General", "Academic", "Technical", "Business"])
                    st.markdown(f"<small><em>Tailored for {audience.lower()} readers.</em></small>", unsafe_allow_html=True)
                
                # Additional options
                with st.expander("Advanced Options"):
                    focus_area = st.multiselect("Focus Areas:", 
                                              ["Main ideas", "Technical details", "Critical analysis", 
                                               "Historical context", "Future implications"],
                                              default=["Main ideas"])
                    
                    tone = st.select_slider("Tone:", 
                                          options=["Formal", "Balanced", "Conversational"],
                                          value="Balanced")
                
                if st.button("Generate Summary", key="gen_summary"):
                    with st.spinner("Generating summary... This may take a moment"):
                        try:
                            # Limit text for API
                            if len(text_to_summarize) > 6000:
                                processing_text = text_to_summarize[:6000]
                                st.info("Document is very large. Summarizing first 6,000 characters.")
                            else:
                                processing_text = text_to_summarize
                                
                            # Build a more detailed prompt
                            focus_text = ", ".join(focus_area)
                            summary_prompt = f"""Summarize the following text in a {summary_type.lower()} format for a {audience.lower()} audience. 
                            Focus on {focus_text} and use a {tone.lower()} tone:
                            
                            {processing_text}
                            """
                            
                            summary = query_groq(processing_text, summary_prompt)
                            
                            # Display summary in a nice card
                            st.markdown("### 📋 Summary")
                            st.markdown(f"""
                            <div style="background-color: {'#2d2d2d' if theme == 'Dark' else '#f8f9fa'}; 
                                       padding: 25px; 
                                       border-radius: 10px; 
                                       border-left: 5px solid #4d4dff;
                                       margin: 20px 0;">
                                <div style="color: {'#e0e0e0' if theme == 'Dark' else '#333'};">
                                    {summary}
                                </div>
                                <div style="margin-top: 20px; font-style: italic; font-size: 0.9rem; color: {'#bbb' if theme == 'Dark' else '#666'};">
                                    {summary_type} summary for {audience} audience
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Download option
                            st.markdown(get_download_link(summary, 
                                                       f"{uploaded_file.name.split('.')[0]}_summary.txt", 
                                                       "📥 Download Summary"), 
                                       unsafe_allow_html=True)
                            
                            # Copy to clipboard option - FIXED VERSION
                            summary_js_safe = summary.replace('`', "'").replace('"', '\\"').replace('\n', '\\n')
                            st.markdown(f"""
                            <div style="margin-top: 20px;">
                                <button onclick="navigator.clipboard.writeText(\"{summary_js_safe}\");"
                                        style="background-color: #6c757d; color: white; border: none; 
                                               padding: 10px 15px; border-radius: 5px; cursor: pointer;">
                                    📋 Copy to Clipboard
                                </button>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error generating summary: {str(e)}")
                            st.info("Try using a shorter document or simplify the text.")
            else:
                st.error(f"Could not process document: {text_to_summarize}")
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

# Document Classification - Using the imported module
elif option == "Document Classification":
    try:
        render_document_classification()
    except Exception as e:
        st.error(f"Error in document classification: {str(e)}")
        st.info("Make sure classification models are properly trained and available.")

# Voice Interaction (future feature)
elif option == "Voice Interaction":
    st.markdown("<div class='subheader'>🎤 Voice Interaction</div>", unsafe_allow_html=True)
    
    st.markdown("""
    Voice interaction allows you to speak to your documents and get spoken responses.
    This feature will enable a more natural way to interact with document content.
    """)
    
    # Feature description with visual appeal
    col1, col2 = st.columns([3, 2])
    
    with col1:
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
    
    with col2:
        # Decorative microphone animation
        st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 200px;">
            <div style="font-size: 5rem; margin-bottom: 20px;" class="loading-animation">🎤</div>
            <div style="text-align: center; color: #6c757d;">Voice Feature Coming Soon</div>
        </div>
        """, unsafe_allow_html=True)

# Footer and sidebar information
st.sidebar.markdown("---")
st.sidebar.info("Developed using Streamlit & AI Models 🚀")
st.sidebar.info("Developed by: Bit-By-Bit🎶")

# Add helpful resources
with st.sidebar.expander("📚 Help & Resources"):
    st.markdown("""
    - [User Guide](#)
    - [API Documentation](https://console.groq.com/docs/api-reference#chat-create)
    - [Report a Bug](#)
    """)

# System status with visual indicators
with st.sidebar.expander("🔧 System Status"):
    status_items = [
        {"name": "Plagiarism Detection", "status": "Online"},
        {"name": "PDF Text Extraction", "status": "Online"},
        {"name": "Vector Database", "status": "Online"},
        {"name": "Groq API Connection", "status": "Online"},
        {"name": "Document Classification", "status": "Online"}
    ]
    
    for item in status_items:
        if item["status"] == "Online":
            icon = "✅"
            color = "green"
        else:
            icon = "⚠️"
            color = "orange"
            
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="color: {color}; margin-right: 8px;">{icon}</div>
            <div style="flex-grow: 1;">{item['name']}</div>
            <div style="color: {color}; font-weight: bold;">{item['status']}</div>
        </div>
        """, unsafe_allow_html=True)
