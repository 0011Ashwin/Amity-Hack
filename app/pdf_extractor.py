import pdfplumber
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
import os
from dotenv import load_dotenv
import groq
from langchain_community.vectorstores import FAISS  # ✅ Updated FAISS Import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Load API Key Properly
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("❌ Groq API key is missing! Set GROQ_API_KEY in .env file.")

# ✅ Initialize Groq Client
groq_client = groq.Client(api_key=groq_api_key)

# ✅ Initialize Vector Database
vector_db = None

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
                    img = page.to_image().original  # Convert page to image
                    ocr_text = pytesseract.image_to_string(img)  # Perform OCR
                    text += ocr_text + "\n"
    except Exception as e:
        return f"Error extracting text: {str(e)}"
    
    return text.strip()

def store_text_in_vector_db(text):
    """Split text into chunks and store in a vector database using HuggingFace embeddings."""
    global vector_db
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_text(text)

    # ✅ Updated FAISS Import & Using HuggingFace Embeddings
    vector_db = FAISS.from_texts(text_chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    return "Text stored in vector DB successfully."

def retrieve_relevant_text(query):
    """Retrieve relevant text chunks from the vector database."""
    if vector_db is None:
        return None  # No data in the vector DB yet
    retrieved_chunks = vector_db.similarity_search(query, k=5)  # Get top 5 relevant chunks
    return " ".join([chunk.page_content for chunk in retrieved_chunks])

def query_groq(text, question):
    """Send extracted text to Groq API and get a response using Llama3 or Mixtral."""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Change to "mixtral-8x7b" if needed
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": f"Context: {text}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content if response.choices else "No response available."

def hybrid_query_pipeline(pdf_path, question):
    """Full pipeline: extract text, store in DB, retrieve, and query LLM."""
    extracted_text = extract_text_from_pdf(pdf_path)
    store_text_in_vector_db(extracted_text)  # Store in vector DB

    relevant_text = retrieve_relevant_text(question)  # Get relevant context
    if relevant_text:  # If relevant text is found in the database
        return query_groq(relevant_text, question)
    else:  # Otherwise, send the full text to LLM
        return query_groq(extracted_text, question)

# ✅ Example Usage
if __name__ == "__main__":
    pdf_path = "data/pdf-1.pdf"
    question = "Summarize the document."
    
    print("\nExtracting and Processing PDF...")
    answer = hybrid_query_pipeline(pdf_path, question)
    
    print("\n💡 AI Response:")
    print(answer)


# 3rd code 

# import streamlit as st
# import speech_recognition as sr
# import pdfplumber
# import pytesseract
# import fitz  # PyMuPDF
# from PIL import Image
# import os
# from dotenv import load_dotenv
# import groq
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from gtts import gTTS
# from pydub import AudioSegment
# from pydub.playback import play

# def text_to_speech(text):
#     """Convert text to speech and play audio."""
#     tts = gTTS(text=text, lang="en")
#     audio_path = "response_audio.mp3"
#     tts.save(audio_path)
#     return audio_path

# def speech_to_text():
#     """Capture voice input and convert to text."""
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         st.info("🎙️ Speak now...")
#         try:
#             audio = recognizer.listen(source, timeout=5)
#             text = recognizer.recognize_google(audio)
#             return text
#         except sr.UnknownValueError:
#             return "Could not understand audio."
#         except sr.RequestError:
#             return "Speech recognition service is unavailable."

# # ✅ Load API Key Properly
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")

# if not groq_api_key:
#     st.error("❌ Groq API key is missing! Set GROQ_API_KEY in .env file.")
#     st.stop()

# # ✅ Initialize Groq Client
# groq_client = groq.Client(api_key=groq_api_key)

# # ✅ Initialize Vector Database
# vector_db = None

# def extract_text_from_pdf(pdf_path):
#     """Extract text from a given PDF file, including OCR for scanned pages."""
#     text = ""
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 extracted_text = page.extract_text()
#                 if extracted_text:
#                     text += extracted_text + "\n"
#                 else:
#                     img = page.to_image().original
#                     ocr_text = pytesseract.image_to_string(img)
#                     text += ocr_text + "\n"
#     except Exception as e:
#         return f"Error extracting text: {str(e)}"
    
#     return text.strip()

# def store_text_in_vector_db(text):
#     """Split text into chunks and store in a vector database using HuggingFace embeddings."""
#     global vector_db
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     text_chunks = text_splitter.split_text(text)
#     vector_db = FAISS.from_texts(text_chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
#     return "Text stored in vector DB successfully."

# def retrieve_relevant_text(query):
#     """Retrieve relevant text chunks from the vector database."""
#     if vector_db is None:
#         return None
#     retrieved_chunks = vector_db.similarity_search(query, k=5)
#     return " ".join([chunk.page_content for chunk in retrieved_chunks])

# def query_groq(text, question):
#     """Send extracted text to Groq API and get a response using Llama3 or Mixtral."""
#     response = groq_client.chat.completions.create(
#         model="llama-3.3-70b-versatile",
#         messages=[
#             {"role": "system", "content": "You are an AI assistant."},
#             {"role": "user", "content": f"Context: {text}\n\nQuestion: {question}"}
#         ]
#     )
#     return response.choices[0].message.content if response.choices else "No response available."

# # ✅ Streamlit UI
# st.title("📄 PDF AI Query Tool with Voice Interaction")

# uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
# if uploaded_file:
#     pdf_path = f"temp_{uploaded_file.name}"
#     with open(pdf_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
    
#     st.success("✅ File uploaded successfully!")
    
#     extracted_text = extract_text_from_pdf(pdf_path)
#     st.subheader("Extracted Text:")
#     st.text_area("", extracted_text, height=300)
    
#     if st.button("Store Text in Vector DB"):
#         store_text_in_vector_db(extracted_text)
#         st.success("✅ Text stored successfully!")
    
#     question = st.text_input("Ask a question about the document:")
    
#     # 🎙️ Voice Input Button
#     if st.button("🎤 Speak your question"):
#         question = speech_to_text()
#         st.write(f"🗣️ You said: {question}")
    
#     if question:
#         answer = query_groq(extracted_text, question)
#         st.subheader("💡 AI Response:")
#         st.write(answer)

#         # Convert response to speech
#         audio_file = text_to_speech(answer)

#         # Play audio in Streamlit
#         st.audio(audio_file, format="audio/mp3")

