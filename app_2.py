# # Streamlit app provide the user interface to show-case 
# import streamlit as st
# import os
# from app.plagiarism_detector import get_similarity
# # from app.document_classifier import classify_document
# from app.pdf_extractor import extract_text_from_pdf
# from app.query_engine import query_pdf


# # Streamlit UI
# st.title("📄 Document Intelligence System")
# st.write("Welcome to the Document Intelligence System! 🚀")
# # Function sidebar 
# st.sidebar.header("Choose a Function:")
# option = st.sidebar.radio("Select a feature:",
#                          ["Plagiarism Detection",
#                         "Document Classification",
#                         "PDF Text Extraction",
#                         "Query PDF"])


# # Plagiarism Detection
# if option == "Plagiarism Detection":
#     st.subheader("🔍 Check Plagiarism Between Two Texts")
#     text1 = st.text_area("Enter First Text:")
#     text2 = st.text_area("Enter Second Text:")
#     if st.button("Check Plagiarism"):
#         if text1 and text2:
#             similarity = get_similarity(text1, text2)
#             st.success(f"Similarity Score: {similarity:.2f}%")
#         else:
#             st.warning("Please enter both texts to compare.")


# # elif option == "Document Classification":
# #     st.subheader("📑 Classify Document into Financial, Healthcare, or Legal")
# #     doc_text = st.text_area("Enter Document Text:")
# #     if st.button("Classify Document"):
# #         if doc_text:
# #             category = classify_document(doc_text)
# #             st.success(f"Predicted Category: {category}")
# #         else:
# #             st.warning("Please enter text for classification.")

# # PDF Text Extraction
# elif option == "PDF Text Extraction":
#     st.subheader("📄 Extract Text from a PDF File")
#     uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])
#     if uploaded_file is not None:
#         save_path = f"temp_{uploaded_file.name}"
#         with open(save_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
#         extracted_text = extract_text_from_pdf(save_path)
#         st.text_area("Extracted Text:", extracted_text, height=300)
#         os.remove(save_path)  # Clean up

# # Query PDF
# elif option == "Query PDF":
#     st.subheader("🤖 Ask Questions from Extracted PDF Text")
#     pdf_text = st.text_area("Enter Extracted Text:")
#     question = st.text_input("Enter Your Question:")
#     if st.button("Get Answer"):
#         if pdf_text and question:
#             answer = query_pdf(pdf_text, question)
#             st.success(f"Answer: {answer}")
#         else:
#             st.warning("Please provide both text and question.")

# st.sidebar.info("Developed using Streamlit & AI Models 🚀")
# st.sidebar.info("Developed by : @Ashwin Mehta🎶")

# Streamlit app provide the user interface to show-case 

import streamlit as st
import os
import tempfile
from app.plagiarism_detector import get_similarity
from app.pdf_extractor import extract_text_from_pdf, process_text_with_gemini
from app.query_engine import query_pdf

# Streamlit UI
st.title("📄 Document Intelligence System")
st.write("Welcome to the Document Intelligence System! 🚀")

# Function sidebar
st.sidebar.header("Choose a Function:")
option = st.sidebar.radio("Select a feature:",
                         ["Plagiarism Detection",
                          "Document Classification",
                          "PDF Text Extraction",
                          "Query PDF"])

# Plagiarism Detection
if option == "Plagiarism Detection":
    st.subheader("🔍 Check Plagiarism Between Two Texts")
    text1 = st.text_area("Enter First Text:")
    text2 = st.text_area("Enter Second Text:")
    if st.button("Check Plagiarism"):
        if text1 and text2:
            similarity = get_similarity(text1, text2)
            st.success(f"Similarity Score: {similarity:.2f}%")
        else:
            st.warning("Please enter both texts to compare.")

# PDF Text Extraction
elif option == "PDF Text Extraction":
    st.subheader("📄 Extract and Process Text from a PDF File")
    uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.getbuffer())
            save_path = temp_pdf.name
        
        st.info("Extracting text from PDF...")
        extracted_text = extract_text_from_pdf(save_path)
        st.text_area("Extracted Text:", extracted_text, height=300)
        
        if st.button("Enhance with AI"):
            st.info("Processing text with AI...")
            ai_text = process_text_with_gemini(extracted_text)
            st.text_area("AI-Enhanced Text:", ai_text, height=300)
        
        os.remove(save_path)  # Clean up

# Query PDF
elif option == "Query PDF":
    st.subheader("🤖 Ask Questions from Extracted PDF Text")
    pdf_text = st.text_area("Enter Extracted Text:")
    question = st.text_input("Enter Your Question:")
    if st.button("Get Answer"):
        if pdf_text and question:
            answer = query_pdf(pdf_text, question)
            st.success(f"Answer: {answer}")
        else:
            st.warning("Please provide both text and question.")

st.sidebar.info("Developed using Streamlit & AI Models 🚀")
st.sidebar.info("Developed by : @Ashwin Mehta🎶")
