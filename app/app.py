from flask import Flask, render_template, request
import os
from plagiarism_detector import get_similarity
from pdf_extractor import extract_text_from_pdf  # Added import
# from pdf_extractor import extract_text_from_pdf
from query_engine import query_pdf

# flask app 
app = Flask(__name__)

# app route which render to index.html
@app.route('/')
def home():
    return render_template('index.html')

# app-route of plagirarism
@app.route('/plagiarism', methods=['GET', 'POST'])
def plagiarism():
    similarity = None
    if request.method == 'POST':
        text1 = request.form.get('text1')
        text2 = request.form.get('text2')
        if text1 and text2:
            similarity = get_similarity(text1, text2)
    return render_template('plagiarism.html', similarity=similarity)

# # app-route of pdf_extraction
@app.route('/pdf_extraction', methods=['GET', 'POST'])
def pdf_extraction():
    extracted_text = None
    if request.method == 'POST':
        pdf_file = request.files.get('pdf_file')
        if pdf_file:
            save_path = f"temp_{pdf_file.filename}"
            pdf_file.save(save_path)
            extracted_text = extract_text_from_pdf(save_path)
            os.remove(save_path)
    return render_template('pdf_extraction.html', extracted_text=extracted_text)

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
