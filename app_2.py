


import streamlit as st
import os
import tempfile
import joblib
from app.plagiarism_detector import get_similarity
from app.pdf_extractor import extract_text_from_pdf, store_text_in_vector_db, query_groq
from app.query_engine import query_pdf



# Streamlit UI
st.title("📄 Document Intelligence System")
st.write("Welcome to the Document Intelligence System! 🚀")

# Function sidebar
st.sidebar.header("Choose a Function:")
option = st.sidebar.radio("Select a feature:",
                         ["Plagiarism Detection",
                          "PDF Text Extraction"])


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
    st.subheader("📄 Query with Hybrid Model")
    uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.getbuffer())
            save_path = temp_pdf.name
        
        st.info("Extracting text from PDF...")
        extracted_text = extract_text_from_pdf(save_path)
        st.text_area("Extracted Text:", extracted_text, height=300)

        if st.button("Store in Vector DB"):
            store_text_in_vector_db(extracted_text)
            st.success("✅ Text stored successfully!")

        question = st.text_input("Ask a question about the document:")
        if question:
            answer = query_groq(extracted_text, question)
            st.subheader("💡 AI Response:")
            st.write(answer)

        os.remove(save_path)  # Clean up

# Query PDF
# elif option == "Q&A with prompt":
#     st.subheader("🤖 Ask Questions from Expert")
#     pdf_text = st.text_area("Enter Your Prompt:")
#     question = st.text_input("Enter Your Question:")
#     if st.button("Get Answer"):
#         if pdf_text and question:
#             answer = query_pdf(pdf_text, question)
#             st.success(f"Answer: {answer}")
#         else:
#             st.warning("Please provide both text and question.")

st.sidebar.info("Developed using Streamlit & AI Models 🚀")
st.sidebar.info("Developed by : @Ashwin Mehta🎶")

# Give me overall details for this paper and how will i study to got good marks
# what do you think What are important questions asked by teacher in exam 
# from app.pdf_extractor import extract_text_from_pdf, process_text_with_gemini