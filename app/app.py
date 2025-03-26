from flask import Flask, render_template, request, jsonify, send_file, url_for, redirect
import os
import tempfile
import joblib
import base64
import json
import io
from plagiarism_detector import get_similarity
from pdf_extractor import extract_text_from_pdf, store_text_in_vector_db, query_groq
from query_engine import query_pdf

# Import document classification functionality
from document_classifier import classify_document, clean_text

# flask app 
app = Flask(__name__)

# Add jinja2 filter for nl2br (new line to <br>)
@app.template_filter('nl2br')
def nl2br_filter(s):
    if s:
        return s.replace('\n', '<br>')
    return s

# app route which render to index.html
@app.route('/')
def home():
    return render_template('index.html')

# app-route of plagirarism
@app.route('/plagiarism', methods=['GET', 'POST'])
def plagiarism():
    similarity = None
    if request.method == 'POST':
        if 'file1' in request.files and request.files['file1'].filename:
            # Handle file upload for text 1
            file1 = request.files['file1']
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + file1.filename.split('.')[-1]) as temp_file:
                file1.save(temp_file.name)
                save_path = temp_file.name
                
                if file1.filename.endswith('.pdf'):
                    text1 = extract_text_from_pdf(save_path)
                else:
                    with open(save_path, 'r', encoding='utf-8') as f:
                        text1 = f.read()
                os.remove(save_path)
        else:
            # Get text from form input
            text1 = request.form.get('text1', '')
            
        if 'file2' in request.files and request.files['file2'].filename:
            # Handle file upload for text 2
            file2 = request.files['file2']
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + file2.filename.split('.')[-1]) as temp_file:
                file2.save(temp_file.name)
                save_path = temp_file.name
                
                if file2.filename.endswith('.pdf'):
                    text2 = extract_text_from_pdf(save_path)
                else:
                    with open(save_path, 'r', encoding='utf-8') as f:
                        text2 = f.read()
                os.remove(save_path)
        else:
            # Get text from form input
            text2 = request.form.get('text2', '')
            
        if text1 and text2:
            similarity = get_similarity(text1, text2)
            
            # Determine similarity level message
            if similarity < 20:
                similarity_level = "Low similarity detected. Documents appear to be original."
                level_class = "success"
            elif similarity < 40:
                similarity_level = "Moderate similarity detected. Some common phrases or ideas may be present."
                level_class = "info"
            elif similarity < 60:
                similarity_level = "Significant similarity detected. Documents share substantial content."
                level_class = "warning"
            else:
                similarity_level = "High similarity detected! Documents may be plagiarized."
                level_class = "danger"
                
            return render_template('plagiarism.html', 
                                  similarity=f"{similarity:.2f}", 
                                  similarity_level=similarity_level,
                                  level_class=level_class,
                                  text1=text1,
                                  text2=text2)
    
    return render_template('plagiarism.html', similarity=None)

# # app-route of pdf_extraction
@app.route('/pdf_extraction', methods=['GET', 'POST'])
def pdf_extraction():
    extracted_text = None
    answer = None
    
    if request.method == 'POST':
        if 'pdf_file' in request.files and request.files['pdf_file'].filename:
            pdf_file = request.files['pdf_file']
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                pdf_file.save(temp_pdf.name)
                save_path = temp_pdf.name
                
            extracted_text = extract_text_from_pdf(save_path)
            os.remove(save_path)
            
            # Check if there's also a question to answer
            question = request.form.get('question')
            if question:
                answer = query_groq(extracted_text, question)
    
    return render_template('pdf_extraction.html', extracted_text=extracted_text, answer=answer)

# app-route of query_pdf 
@app.route('/query_pdf', methods=['GET', 'POST'])
def query_pdf_page():
    answer = None
    if request.method == 'POST':
        pdf_text = request.form.get('pdf_text')
        question = request.form.get('question')
        if pdf_text and question:
            answer = query_pdf(pdf_text, question)
    return render_template('query_pdf.html', answer=answer)

@app.route('/document_summarization', methods=['GET', 'POST'])
def document_summarization():
    summary = None
    document_text = None
    
    if request.method == 'POST':
        if 'document_file' in request.files and request.files['document_file'].filename:
            doc_file = request.files['document_file']
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + doc_file.filename.split('.')[-1]) as temp_file:
                doc_file.save(temp_file.name)
                save_path = temp_file.name
                
                if doc_file.filename.endswith('.pdf'):
                    document_text = extract_text_from_pdf(save_path)
                else:
                    with open(save_path, 'r', encoding='utf-8') as f:
                        document_text = f.read()
                os.remove(save_path)
                
                # Generate summary if requested
                if 'summarize' in request.form:
                    summary_type = request.form.get('summary_type', 'Brief')
                    audience = request.form.get('audience', 'General')
                    
                    summary_prompt = f"Summarize the following text in a {summary_type.lower()} format for a {audience.lower()} audience:\n\n{document_text}"
                    summary = query_groq(document_text, summary_prompt)
    
    return render_template('document_summarization.html', 
                          document_text=document_text, 
                          summary=summary)

@app.route('/document_classification', methods=['GET', 'POST'])
def document_classification():
    document_text = None
    classification_result = None
    confidence = None
    probabilities = None
    
    if request.method == 'POST' and 'document_file' in request.files:
        doc_file = request.files['document_file']
        if doc_file.filename:
            # Use a safer approach for handling files
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + doc_file.filename.split('.')[-1]) as temp_file:
                doc_file.save(temp_file.name)
                save_path = temp_file.name
                
                # Extract text based on file type
                if doc_file.filename.endswith('.pdf'):
                    document_text = extract_text_from_pdf(save_path)
                elif doc_file.filename.endswith('.docx'):
                    try:
                        import docx
                        doc = docx.Document(save_path)
                        document_text = "\n".join([para.text for para in doc.paragraphs])
                    except ImportError:
                        document_text = "DOCX extraction requires python-docx package."
                else:
                    # Only try UTF-8 decoding for text files
                    try:
                        with open(save_path, 'r', encoding='utf-8') as f:
                            document_text = f.read()
                    except UnicodeDecodeError:
                        document_text = "Cannot decode file. It may be a binary file type."
                
                os.remove(save_path)
                
                # Classify document
                if document_text:
                    try:
                        # Load models
                        vectorizer_path = "document_classifier/vectorizer.pkl"
                        classifier_path = "document_classifier/text_classifier.pkl"
                        
                        vectorizer = joblib.load(vectorizer_path)
                        classifier = joblib.load(classifier_path)
                        
                        # Clean and classify text
                        clean_document_text = clean_text(document_text)
                        features = vectorizer.transform([clean_document_text])
                        
                        # Get prediction and probabilities
                        prediction = classifier.predict(features)[0]
                        probs = classifier.predict_proba(features)[0]
                        
                        # Get class names
                        if hasattr(classifier, 'classes_'):
                            categories = classifier.classes_
                        else:
                            categories = ["healthcare", "legal", "financial"]
                        
                        # Create mapping for display names
                        display_names = {
                            "healthcare": "Healthcare",
                            "health": "Healthcare",
                            "legal": "Legal",
                            "financial": "Financial",
                            "finance": "Financial"
                        }
                        
                        # Prepare results
                        predicted_category = str(prediction)
                        display_category = display_names.get(predicted_category.lower(), predicted_category)
                        confidence = f"{probs.max()*100:.2f}"
                        
                        # Prepare probabilities for display
                        prob_data = []
                        for i, cat in enumerate(categories):
                            display_cat = display_names.get(str(cat).lower(), str(cat))
                            prob_data.append({
                                "category": display_cat,
                                "probability": f"{probs[i]*100:.2f}"
                            })
                        
                        # Sort by probability (descending)
                        prob_data = sorted(prob_data, key=lambda x: float(x["probability"]), reverse=True)
                        
                        # Format for JSON
                        probabilities = json.dumps(prob_data)
                        classification_result = display_category
                        
                    except Exception as e:
                        classification_result = None
                        document_text += f"\n\nError during classification: {str(e)}"
    
    return render_template('document_classification.html',
                          document_text=document_text,
                          classification_result=classification_result,
                          confidence=confidence,
                          probabilities=probabilities)

@app.route('/download_text', methods=['POST'])
def download_text():
    text = request.form.get('text')
    filename = request.form.get('filename', 'document.txt')
    
    if not text:
        return redirect(request.referrer)
    
    # Create in-memory file
    text_io = io.BytesIO(text.encode('utf-8'))
    text_io.seek(0)
    
    return send_file(
        text_io,
        as_attachment=True,
        download_name=filename,
        mimetype='text/plain'
    )

if __name__ == '__main__':
    app.run(debug=True)


























# from flask import Flask, render_template, request
# import os
# import tempfile
# from plagiarism_detector import get_similarity
# from pdf_extractor import extract_text_from_pdf, process_text_with_gemini
# from query_engine import query_pdf

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/plagiarism', methods=['GET', 'POST'])
# def plagiarism():
#     similarity = None
#     if request.method == 'POST':
#         text1 = request.form.get('text1')
#         text2 = request.form.get('text2')
#         if text1 and text2:
#             similarity = get_similarity(text1, text2)
#     return render_template('plagiarism.html', similarity=similarity)

# @app.route('/pdf_extraction', methods=['GET', 'POST'])
# def pdf_extraction():
#     extracted_text = None
#     enhanced_text = None
#     error = None

#     if request.method == 'POST':
#         if 'pdf_file' in request.files and request.files['pdf_file'].filename != "":
#             pdf_file = request.files['pdf_file']
#             try:
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
#                     pdf_file.save(temp_pdf.name)
#                     save_path = temp_pdf.name
                
#                 extracted_text = extract_text_from_pdf(save_path)
#                 os.remove(save_path)  # Cleanup
#             except Exception as e:
#                 error = f"Error processing file: {str(e)}"
        
#         # Check if AI enhancement was requested
#         if 'enhance' in request.form and extracted_text:
#             try:
#                 enhanced_text = process_text_with_gemini(extracted_text)
#             except Exception as e:
#                 error = f"Error enhancing text: {str(e)}"
    
#     return render_template('pdf_extraction.html', extracted_text=extracted_text, enhanced_text=enhanced_text, error=error)

# @app.route('/query_pdf', methods=['GET', 'POST'])
# def query_pdf_page():
#     answer = None
#     if request.method == 'POST':
#         pdf_text = request.form.get('pdf_text')
#         question = request.form.get('question')
#         if pdf_text and question:
#             answer = query_pdf(pdf_text, question)
#     return render_template('query_pdf.html', answer=answer)

# if __name__ == '__main__':
#     app.run(debug=True)
