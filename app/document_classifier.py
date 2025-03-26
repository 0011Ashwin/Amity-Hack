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
    if not isinstance(text, str):
        return ""
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
    try:
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
    except Exception as e:
        return f"Error processing document: {str(e)}", ""

def render_document_classification():
    """Render document classification UI in Streamlit"""
    st.markdown("<div class='subheader'>📋 Document Classification</div>", unsafe_allow_html=True)
    
    st.markdown("""
    This tool analyzes documents and classifies them into categories: healthcare, legal, or financial.
    Upload a document to see its classification and confidence level.
    """)
    
    # File uploader with enhanced UI
    uploaded_file = st.file_uploader("Upload a Document", type=["pdf", "txt", "docx"], key="classification_file")
    
    if uploaded_file is not None:
        with st.spinner("Processing your document..."):
            try:
                # Get document text
                doc_text, doc_text_cleaned = classify_document(uploaded_file)
                
                if doc_text.startswith("Error"):
                    st.error(doc_text)
                    return
                
                # Display extracted text
                with st.expander("View Document Text"):
                    # Truncate text if too long
                    display_text = doc_text
                    if len(doc_text) > 5000:
                        display_text = doc_text[:5000] + "... (truncated for display)"
                        st.info("Document is large. Displaying first 5,000 characters.")
                    
                    st.text_area("Content:", display_text, height=200)
                
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
                            st.write("Model classes:", [str(c) for c in classifier.classes_])
                    
                    # Transform text using vectorizer
                    features = vectorizer.transform([doc_text_cleaned])
                    
                    # Get the actual categories from the model
                    if hasattr(classifier, 'classes_'):
                        categories = [str(c) for c in classifier.classes_]
                    else:
                        # Fallback to the categories from the training script
                        categories = ["healthcare", "legal", "financial"]
                    
                    # Predict category
                    try:
                        prediction_raw = classifier.predict(features)[0]
                        # Handle different prediction types
                        if isinstance(prediction_raw, (np.integer, int)):
                            prediction_idx = int(prediction_raw)  # Convert numpy.int64 to Python int
                            predicted_category = categories[prediction_idx]
                        else:
                            # If prediction is already the category name
                            predicted_category = str(prediction_raw)
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        st.error(f"Prediction type: {type(prediction_raw)}")
                        predicted_category = "Unknown"
                    
                    # Get probabilities
                    try:
                        raw_probabilities = classifier.predict_proba(features)[0]
                        # Convert to Python floats to avoid numpy type issues
                        probabilities = [float(p) for p in raw_probabilities]
                    except Exception as e:
                        st.error(f"Probability calculation error: {str(e)}")
                        # Create dummy equal probabilities as fallback
                        probabilities = [1.0/len(categories)] * len(categories)
                    
                    # Display name mapping
                    display_names = {
                        "healthcare": "Healthcare",
                        "health": "Healthcare",
                        "legal": "Legal",
                        "financial": "Financial",
                        "finance": "Financial"
                    }
                    
                    # Get display name
                    display_category = display_names.get(predicted_category.lower(), predicted_category)
                    
                    # Icons mapping
                    icon_map = {
                        "Healthcare": "🏥",
                        "Legal": "⚖️",
                        "Financial": "💰"
                    }
                    
                    # Get icon
                    icon = icon_map.get(display_category, "📄")
                    
                    # Calculate max confidence
                    max_prob = float(max(probabilities)) * 100
                    
                    # Theme detection for styling
                    theme = "Dark" if st.session_state.get("_theme", {}).get("base", "") == "dark" else "Light"
                    card_bg = "#2d2d2d" if theme == "Dark" else "#f8f9fa"
                    text_color = "#e0e0e0" if theme == "Dark" else "#333"
                    
                    # Display classification results with enhanced UI
                    st.markdown("### Classification Results")
                    st.markdown(f"""
                    <div style="background-color: {card_bg}; 
                               padding: 25px; 
                               border-radius: 10px; 
                               border-left: 5px solid #4d4dff;
                               margin: 20px 0;
                               text-align: center;">
                        <div style="font-size: 4rem; margin-bottom: 15px;">{icon}</div>
                        <div style="font-size: 2rem; font-weight: bold; margin: 10px 0; color: {text_color};">
                            {display_category}
                        </div>
                        <div style="font-size: 1.2rem; color: #666; margin-top: 15px;">
                            Confidence: <span style="font-weight: bold;">{max_prob:.2f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display confidence breakdown with animation
                    st.markdown("### Confidence Breakdown")
                    
                    # Prepare data for visualization
                    display_categories = [display_names.get(cat.lower(), cat) for cat in categories]
                    display_icons = [icon_map.get(cat, "📄") for cat in display_categories]
                    
                    # Create DataFrame with probabilities and icons
                    probs_df = pd.DataFrame({
                        'Category': display_categories,
                        'Confidence': [float(p) * 100 for p in probabilities],
                        'Icon': display_icons
                    })
                    
                    # Sort by confidence
                    probs_df = probs_df.sort_values('Confidence', ascending=False)
                    
                    # Display bar chart for confidence values
                    chart_df = probs_df[['Category', 'Confidence']].set_index('Category')
                    st.bar_chart(chart_df)
                    
                    # Custom progress bars for each category
                    for i, row in probs_df.iterrows():
                        category = row['Category']
                        prob = row['Confidence']
                        icon_symbol = row['Icon']
                        
                        # Determine color based on probability
                        if category == display_category:
                            bar_color = "#4d4dff"  # Highlight the predicted category
                        else:
                            bar_color = "#6c757d"  # Gray for other categories
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 15px;">
                            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                <div style="margin-right: 10px; font-size: 1.2rem;">{icon_symbol}</div>
                                <div style="font-weight: bold;">{category}</div>
                                <div style="margin-left: auto; font-weight: bold;">{prob:.2f}%</div>
                            </div>
                            <div style="height: 10px; background-color: #e9ecef; border-radius: 5px; overflow: hidden;">
                                <div style="width: {prob}%; height: 100%; background-color: {bar_color};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Document traits analysis based on classification
                    st.markdown("### Document Traits Analysis")
                    
                    # Set up traits based on category
                    if display_category == "Healthcare":
                        traits = [
                            "Contains medical terminology",
                            "May include patient information",
                            "Likely has health-related procedures or diagnoses",
                            "May contain healthcare provider information"
                        ]
                        trait_icon = "🏥"
                    elif display_category == "Legal":
                        traits = [
                            "Contains legal terminology",
                            "May include case citations or references",
                            "Likely has formal legal structure",
                            "May contain contractual elements or legal agreements"
                        ]
                        trait_icon = "⚖️"
                    else:  # Financial
                        traits = [
                            "Contains financial terminology",
                            "May include numerical data and monetary values",
                            "Likely has financial analysis or reporting elements",
                            "May contain investment or accounting information"
                        ]
                        trait_icon = "💰"
                    
                    # Display traits with icons and animation
                    for trait in traits:
                        st.markdown(f"""
                        <div class="card" style="background-color: {card_bg}; padding: 15px; 
                                     border-radius: 8px; margin-bottom: 10px; 
                                     display: flex; align-items: center;">
                            <div style="margin-right: 15px; font-size: 1.2rem;">{trait_icon}</div>
                            <div style="color: {text_color};">{trait}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add recommendations based on document type
                    st.markdown("### Recommendations")
                    
                    if display_category == "Healthcare":
                        recommendations = [
                            "Ensure HIPAA compliance if this document contains patient information.",
                            "Consider medical record retention requirements for this document.",
                            "Verify all medical terms and procedures are accurately represented.",
                            "Check for sensitive patient data that may need special handling."
                        ]
                    elif display_category == "Legal":
                        recommendations = [
                            "Verify legal citations and references for accuracy.",
                            "Consider having a legal professional review this document.",
                            "Check that all legal terms are clearly defined.",
                            "Ensure document complies with relevant jurisdictional requirements."
                        ]
                    else:  # Financial
                        recommendations = [
                            "Check financial calculations and figures for accuracy.",
                            "Consider regulatory compliance requirements for financial documents.",
                            "Verify that financial data is properly formatted and labeled.",
                            "Ensure sensitive financial information is appropriately secured."
                        ]
                    
                    # Display recommendations with animated cards
                    for i, recommendation in enumerate(recommendations):
                        delay = i * 0.2  # Staggered animation delay
                        st.markdown(f"""
                        <div class="card" style="background-color: {card_bg}; 
                                     padding: 15px; border-radius: 8px; 
                                     margin-bottom: 10px; border-left: 3px solid #4d4dff; 
                                     animation-delay: {delay}s;">
                            <div style="display: flex; align-items: center;">
                                <div style="margin-right: 15px; font-size: 1.2rem;">📌</div>
                                <div style="color: {text_color};">{recommendation}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error in classification: {str(e)}")
                    st.warning("Please ensure the classification models are available at the correct paths.")
                    st.info("If you haven't trained document classification models yet, you'll need to create them first.")
                    
                    # Provide information on how to train models
                    with st.expander("How to train document classification models"):
                        st.markdown("""
                        ### Creating Document Classification Models
                        
                        Follow these steps to train your own document classification models:
                        
                        1. **Prepare Your Dataset**
                           - Create a folder structure with categories (healthcare, legal, financial)
                           - Add sample documents to each category folder
                        
                        2. **Install Required Libraries**
                           ```bash
                           pip install scikit-learn pandas numpy joblib PyPDF2 python-docx
                           ```
                        
                        3. **Create a Training Script**
                           ```python
                           from sklearn.feature_extraction.text import TfidfVectorizer
                           from sklearn.linear_model import LogisticRegression
                           import joblib
                           import os
                           
                           # Load documents from categories
                           documents = []
                           labels = []
                           
                           # Assuming a folder structure with category names as folders
                           for category in ["healthcare", "legal", "financial"]:
                               category_path = f"document_classifier/training_data/{category}"
                               for filename in os.listdir(category_path):
                                   with open(os.path.join(category_path, filename), 'r') as file:
                                       text = file.read()
                                       documents.append(text)
                                       labels.append(category)
                           
                           # Create and train vectorizer
                           vectorizer = TfidfVectorizer()
                           X_train_vectorized = vectorizer.fit_transform(documents)
                           
                           # Train model
                           model = LogisticRegression(max_iter=1000)
                           model.fit(X_train_vectorized, labels)
                           
                           # Save model and vectorizer
                           joblib.dump(model, "document_classifier/text_classifier.pkl")
                           joblib.dump(vectorizer, "document_classifier/vectorizer.pkl")
                           ```
                        
                        4. **Run Your Training Script**
                           - Execute the script after setting up your data
                           - Verify the model files are created
                        
                        5. **Test Your Models**
                           - Use the Document Classification feature with sample documents
                           - Adjust your training data if needed to improve accuracy
                        """)
            except Exception as e:
                st.error(f"Document processing error: {str(e)}")
                st.warning("There was an error processing your document. Please try again with a different file.")
