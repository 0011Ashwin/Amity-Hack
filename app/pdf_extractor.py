import pdfplumber
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key from environment variable
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Google Gemini API key is missing. Set GEMINI_API_KEY in .env file.")

genai.configure(api_key=api_key)

def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file, including OCR for scanned pages."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
                else:
                    # Convert scanned page to image and apply OCR
                    img = page.to_image().original  # Ensure correct image extraction
                    ocr_text = pytesseract.image_to_string(img)
                    text += ocr_text + "\n"
    except Exception as e:
        return f"Error extracting text: {str(e)}"
    
    return text.strip()

def process_text_with_gemini(text):
    """Enhance extracted text using Gemini AI for summarization or analysis."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Summarize and analyze this text: {text[:3000]}")
        return response.text if hasattr(response, "text") else "No AI response available."
    except Exception as e:
        return f"Error processing text with AI: {str(e)}"

if __name__ == "__main__":
    pdf_path = "data/sample-1.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)
    print("Raw Extracted Text:")
    print(extracted_text[:5000])  # Print first 5000 characters
    
    print("\nEnhanced AI Processed Text:")
    ai_text = process_text_with_gemini(extracted_text)
    print(ai_text)
