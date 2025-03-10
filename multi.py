from dotenv import load_dotenv
import streamlit as st
import os
import fitz  # PyMuPDF for PDF text extraction
import google.generativeai as genai
import pandas as pd
import pytesseract
from PIL import Image
from docx import Document

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY_2"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def extract_text_from_docx(uploaded_file):
    """Extracts text from an uploaded DOCX file."""
    doc = Document(uploaded_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_txt(uploaded_file):
    """Reads text from a TXT file."""
    return uploaded_file.read().decode("utf-8")

def extract_text_from_excel(uploaded_file):
    """Reads data from Excel file and converts it to a string."""
    df = pd.read_excel(uploaded_file)
    return df.to_string()

def extract_text_from_csv(uploaded_file):
    """Reads data from CSV file and converts it to a string."""
    df = pd.read_csv(uploaded_file)
    return df.to_string()

def extract_text_from_image(uploaded_file):
    """Extracts text from an image using OCR."""
    image = Image.open(uploaded_file)
    return pytesseract.image_to_string(image)

def get_gemini_response(text, prompt):
    """Sends extracted text to Gemini API for processing."""
    response = model.generate_content([prompt, text])
    return response.text

# Streamlit UI
st.set_page_config(page_title="Document Information Extractor", page_icon="📄", layout="centered")
st.header("📄 Multi-Format Document Extractor")

input_prompt = st.text_area("Enter your query about the document:", "Extract key details such as dates, names, totals, and important sections.")
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt", "xlsx", "csv", "jpg", "png"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    submit = st.button("Extract Information")
    
    if submit:
        with st.spinner("Extracting text from document..."):
            file_type = uploaded_file.type
            
            if "pdf" in file_type:
                extracted_text = extract_text_from_pdf(uploaded_file)
            elif "word" in file_type or "docx" in uploaded_file.name:
                extracted_text = extract_text_from_docx(uploaded_file)
            elif "text" in file_type or "txt" in uploaded_file.name:
                extracted_text = extract_text_from_txt(uploaded_file)
            elif "spreadsheet" in file_type or "xlsx" in uploaded_file.name:
                extracted_text = extract_text_from_excel(uploaded_file)
            elif "csv" in file_type:
                extracted_text = extract_text_from_csv(uploaded_file)
            elif "image" in file_type or uploaded_file.name.endswith(("jpg", "png")):
                extracted_text = extract_text_from_image(uploaded_file)
            else:
                extracted_text = "Unsupported file format."
                
        if extracted_text:
            with st.spinner("Processing with Gemini..."):
                response = get_gemini_response(extracted_text, input_prompt)
            
            st.subheader("Extracted Information:")
            st.write(response)
