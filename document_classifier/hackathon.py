# importing libraries 
import os
import re
import joblib
import streamlit as st
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() if page.extract_text() else ""
    return text

# Function to extract text from TXT files
def extract_text_from_txt(txt_file):
    with open(txt_file, "r", encoding="utf-8") as f:
        return f.read()

# Function to extract text from DOCX files
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    return text

# Folder paths for training data
training_data_dir = "training_data"
categories = ["financial", "healthcare", "legal"]

# Prepare training data
documents, labels = [], []

for category in categories:
    category_path = os.path.join(training_data_dir, category)

    if os.path.exists(category_path):
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            text = ""

            # Extract text based on file type
            if file.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            elif file.endswith(".txt"):
                text = extract_text_from_txt(file_path)
            elif file.endswith(".docx"):
                text = extract_text_from_docx(file_path)

            if text.strip():  # Only process non-empty text
                text = clean_text(text)
                documents.append(text)
                labels.append(category)  # Assign category label
            else:
                print(f"⚠️ Skipping {file} (No text extracted)")

# Ensure at least two categories are present before training
if len(set(labels)) < 2:
    raise ValueError("This solver needs samples of at least 2 classes, but only one class found.")

# Convert text data into numerical features
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(documents)

# Train model
model = LogisticRegression()
model.fit(X_train_vectorized, labels)

# Save model and vectorizer
joblib.dump(model, "text_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Streamlit UI
st.title("File Category Classifier")

uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt", "docx"])

if uploaded_file:
    # Save uploaded file temporarily
    temp_path = f"temp.{uploaded_file.name.split('.')[-1]}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text
    text = ""
    if uploaded_file.name.endswith(".pdf"):
        text = extract_text_from_pdf(temp_path)
    elif uploaded_file.name.endswith(".txt"):
        text = extract_text_from_txt(temp_path)
    elif uploaded_file.name.endswith(".docx"):
        text = extract_text_from_docx(temp_path)

    # Classify text
    if text.strip():
        text_cleaned = clean_text(text)
        text_vectorized = vectorizer.transform([text_cleaned])
        prediction = model.predict(text_vectorized)[0]

        st.write(f"### Predicted Category: **{prediction}**")
    else:
        st.error("⚠️ No text extracted. Please try another file.")



