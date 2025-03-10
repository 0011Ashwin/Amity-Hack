import pdfplumber
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import google.generativeai as genai

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
    model = genai.GenerativeModel("gemini-1.5-flash")  # Use the Gemini Pro model
    response = model.generate_content(f"Summarize and analyze this text: {text[:3000]}")
    return response.text if hasattr(response, "text") else "No AI response available."

if __name__ == "__main__":
    pdf_path = "data/pdf-1.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)
    print("Raw Extracted Text:")
    print(extracted_text[:5000])  # Print first 5000 characters
    
    print("\nEnhanced AI Processed Text:")
    ai_text = process_text_with_gemini(extracted_text)
    print(ai_text)
