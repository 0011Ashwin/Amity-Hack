import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import google.generativeai as genai
import os

# Configure Gemini AI API (Replace with your actual API Key)
genai.configure(api_key="AIzaSyBn_WqRupmVpJXloik6wns5pgHWXz1DVBw")

def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file, including OCR for scanned pages."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
            else:
                # Convert scanned page to image and apply OCR
                image = page.to_image()
                ocr_text = pytesseract.image_to_string(image)
                text += ocr_text + "\n"
    return text.strip()

def process_text_with_gemini(text):
    """Enhance extracted text using Gemini AI for summarization or analysis."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"Summarize and analyze this text: {text[:3000]}")
    return response.text if hasattr(response, "text") else "No AI response available."

# Streamlit UI
st.title("📄 PDF Text Extractor & AI Enhancer")
st.write("Upload a PDF file to extract text and optionally enhance it using AI.")

uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])
if uploaded_file is not None:
    save_path = f"temp_{uploaded_file.name}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(save_path)
    st.text_area("Extracted Text:", extracted_text, height=300)
    
    if st.button("Enhance with AI"):
        st.info("Processing text with AI...")
        ai_text = process_text_with_gemini(extracted_text)
        st.text_area("AI-Enhanced Text:", ai_text, height=300)
    
    os.remove(save_path)  # Clean up