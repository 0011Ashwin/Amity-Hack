import os
import re
import joblib
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
from app.pdf_extractor import extract_text_from_pdf

# Function to clean text for classification
def clean_text(text):
    """Clean text for classification"""
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    return text

# Attempt to load DOCX document processing
try:
    import docx
    def extract_text_from_docx(docx_file):
        """Extract text from DOCX files"""
        doc = docx.Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs])
except ImportError:
    def extract_text_from_docx(docx_file):
        """Fallback when python-docx not available"""
        return "DOCX extraction requires python-docx package. Please install it with 'pip install python-docx'."

def classify_document(uploaded_file):
    """Get document text"""
    # Extract text from document
    doc_text = ""
    if uploaded_file.name.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            save_path = temp_file.name
        doc_text = extract_text_from_pdf(save_path)
        os.remove(save_path)
    elif uploaded_file.name.endswith('.docx'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            save_path = temp_file.name
        doc_text = extract_text_from_docx(save_path)
        os.remove(save_path)
    else:
        # Only decode as UTF-8 for text files
        try:
            doc_text = uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            doc_text = "Cannot decode file. It may be a binary file type that we don't support."
    
    # Clean text for better classification
    doc_text_cleaned = clean_text(doc_text)
    
    return doc_text, doc_text_cleaned

def render_document_classification():
    """Render document classification UI in Streamlit"""
    st.markdown("<div class='subheader'>📋 Document Classification</div>", unsafe_allow_html=True)
    
    st.write("Upload a document to classify it into categories: healthcare, legal, or financial.")
    
    uploaded_file = st.file_uploader("Upload a Document", type=["pdf", "txt", "docx"], key="classification_file")
    
    if uploaded_file is not None:
        with st.spinner("Processing your document..."):
            # Get document text
            doc_text, doc_text_cleaned = classify_document(uploaded_file)
            
            # Display extracted text
            with st.expander("View Document Text"):
                st.text_area("Content:", doc_text, height=200)
            
            # Load models for classification
            try:
                # Load trained models
                vectorizer_path = "document_classifier/vectorizer.pkl"
                classifier_path = "document_classifier/text_classifier.pkl"
                
                vectorizer = joblib.load(vectorizer_path)
                classifier = joblib.load(classifier_path)
                
                # Debug model information
                with st.expander("Model Information"):
                    st.write(f"Model type: {type(classifier).__name__}")
                    if hasattr(classifier, 'classes_'):
                        st.write("Model classes:", classifier.classes_)
                
                # Transform text using vectorizer
                features = vectorizer.transform([doc_text_cleaned])
                
                # Get the actual categories from the model
                if hasattr(classifier, 'classes_'):
                    categories = [str(c) for c in classifier.classes_]
                else:
                    # Fallback to the categories from the training script
                    categories = ["healthcare", "legal", "financial"]
                
                # Safely predict category
                try:
                    prediction_idx = classifier.predict(features)[0]
                    # Convert the prediction to a Python primitive
                    if isinstance(prediction_idx, (np.integer, int)):
                        # If prediction is an index
                        prediction_idx = int(prediction_idx)
                        predicted_category = categories[prediction_idx]
                    else:
                        # If prediction is already the category name
                        predicted_category = str(prediction_idx)
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    predicted_category = "Unknown"
                
                # Safely get probabilities
                try:
                    raw_probabilities = classifier.predict_proba(features)[0]
                    # Convert to standard Python floats to avoid numeric type issues
                    probabilities = [float(p) for p in raw_probabilities]
                except Exception as e:
                    st.error(f"Probability calculation error: {str(e)}")
                    # Create dummy probabilities if actual calculation fails
                    probabilities = [0.33, 0.33, 0.34]  # Default equal probabilities
                
                # Display names mapping
                display_names = {
                    "healthcare": "Healthcare",
                    "health": "Healthcare",
                    "legal": "Legal",
                    "financial": "Financial",
                    "finance": "Financial"
                }
                
                # Get a nice display name
                display_category = display_names.get(predicted_category.lower(), predicted_category)
                
                # Icons mapping
                icon_map = {
                    "Healthcare": "🏥",
                    "Legal": "⚖️",
                    "Financial": "💰"
                }
                
                # Get appropriate icon
                icon = icon_map.get(display_category, "📄")
                
                # Calculate confidence - find maximum probability
                max_prob = max(probabilities) * 100
                
                # Display main results
                st.markdown("### Classification Results")
                st.markdown(f"""
                <div style="background-color: #f0f7ff; padding: 20px; border-radius: 10px; text-align: center;">
                    <div style="font-size: 4rem;">{icon}</div>
                    <div style="font-size: 2rem; font-weight: bold; margin: 10px 0;">
                        {display_category}
                    </div>
                    <div style="font-size: 1.2rem; color: #555;">
                        Confidence: {max_prob:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display probability breakdown
                st.markdown("### Confidence Breakdown")
                
                # Prepare data for visualization
                display_categories = [display_names.get(cat.lower(), cat) for cat in categories]
                display_icons = [icon_map.get(cat, "📄") for cat in display_categories]
                
                # Create DataFrame with Python primitive types
                confidence_data = {
                    'Category': display_categories,
                    'Confidence': [p * 100 for p in probabilities],
                }
                
                probs_df = pd.DataFrame(confidence_data)
                
                # Sort by probability
                probs_df = probs_df.sort_values('Confidence', ascending=False)
                
                # Display as bar chart
                st.bar_chart(probs_df.set_index('Category'))
                
                # Show confidence levels in text
                for i, row in probs_df.iterrows():
                    category = row['Category']
                    prob = row['Confidence']
                    icon_symbol = icon_map.get(category, "📄")
                    st.markdown(f"**{category}**: {prob:.2f}% {icon_symbol}")
                
                # Document traits analysis based on classification
                st.markdown("### Document Traits Analysis")
                
                if display_category == "Healthcare":
                    traits = [
                        "Contains medical terminology",
                        "May include patient information",
                        "Likely has health-related procedures or diagnoses",
                        "May contain healthcare provider information"
                    ]
                elif display_category == "Legal":
                    traits = [
                        "Contains legal terminology",
                        "May include case citations or references",
                        "Likely has formal legal structure",
                        "May contain contractual elements or legal agreements"
                    ]
                else:  # Financial
                    traits = [
                        "Contains financial terminology",
                        "May include numerical data and monetary values",
                        "Likely has financial analysis or reporting elements",
                        "May contain investment or accounting information"
                    ]
                
                for trait in traits:
                    st.markdown(f"- {trait}")
                
                # Add recommendations based on document type
                st.markdown("### Recommendations")
                
                if display_category == "Healthcare":
                    st.info("📌 Ensure HIPAA compliance if this document contains patient information.")
                    st.info("📌 Consider medical record retention requirements for this document.")
                elif display_category == "Legal":
                    st.info("📌 Verify legal citations and references for accuracy.")
                    st.info("📌 Consider having a legal professional review this document.")
                else:  # Financial
                    st.info("📌 Check financial calculations and figures for accuracy.")
                    st.info("📌 Consider regulatory compliance requirements for financial documents.")
                
            except Exception as e:
                st.error(f"Error classifying document: {str(e)}")
                st.warning("Please ensure the classification models are available at the correct paths.")
                st.info("If you haven't trained document classification models yet, you'll need to create them first.")
                
                # Provide information on how to train models
                with st.expander("How to train document classification models"):
                    st.markdown("""
                    To create document classification models:
                    
                    1. Prepare a dataset of labeled documents across your categories (healthcare, legal, financial)
                    2. Use scikit-learn to train a classifier:
                    ```python
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.ensemble import RandomForestClassifier
                    import joblib
                    
                    # Assuming documents contains document text and labels contains categories
                    vectorizer = TfidfVectorizer()
                    X_train_vectorized = vectorizer.fit_transform(documents)
                    
                    model = LogisticRegression()  # or RandomForestClassifier()
                    model.fit(X_train_vectorized, labels)
                    
                    # Save model and vectorizer
                    joblib.dump(model, "text_classifier.pkl")
                    joblib.dump(vectorizer, "vectorizer.pkl")
                    ```
                    """) 