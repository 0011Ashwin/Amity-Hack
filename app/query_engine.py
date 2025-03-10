# import google.generativeai as genai
# import os

# # Set API key (ensure it's correct)
# GEMINI_API_KEY = "AIzaSyBn_WqRupmVpJXloik6wns5pgHWXz1DVBw"  # Replace with your actual API key

# # Configure Google Gemini API
# genai.configure(api_key=GEMINI_API_KEY)

# # Function to query Gemini
# def query_pdf(text, question):
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     response = model.generate_content(f"Context: {text}\n\nQuestion: {question}")
#     return response.text  # Ensure you're returning response.text

# # usage case for exmaple 
# # Example usage
# if __name__ == "__main__":
#     sample_text = "I am ashwin mehta and my age is 19yr old."
#     question = "What is person name and his age?"
#     print(f"Answer: {query_pdf(sample_text, question)}")

import google.generativeai as genai
import pdfplumber
import os

# Set API key (ensure it's correct)
GEMINI_API_KEY = "AIzaSyBn_WqRupmVpJXloik6wns5pgHWXz1DVBw"  # Replace with your actual API key

# Configure Google Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def extract_text_from_pdf(pdf_path):
    """Extract text from the uploaded PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text.strip()

def query_pdf(text, question):
    """Ask a question about the extracted PDF text using Google Gemini AI."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"Context: {text}\n\nQuestion: {question}")
    return response.text if hasattr(response, "text") else "No response available."

if __name__ == "__main__":
    pdf_path = input("Enter PDF file path: ")  # Ask user to enter the PDF file path
    
    if os.path.exists(pdf_path):
        extracted_text = extract_text_from_pdf(pdf_path)
        print("\nPDF Text Extracted Successfully!\n")
        
        while True:
            question = input("Ask a question about the document (or type 'exit' to quit): ")
            if question.lower() == "exit":
                print("Exiting...")
                break
            answer = query_pdf(extracted_text, question)
            print(f"\nAnswer: {answer}\n")
    else:
        print("Error: File not found. Please enter a valid PDF path.")
