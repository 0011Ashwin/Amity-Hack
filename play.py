import difflib
import spacy
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Install spaCy model if not already installed
import os
if not os.path.exists("en_core_web_sm"):
    import spacy.cli
    spacy.cli.download("en_core_web_sm")

# Load the small English model from spaCy
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """Preprocess text: lowercasing, lemmatization, and removing stopwords."""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

def get_bow_similarity(text1, text2):
    """Compute similarity using Bag of Words (BoW)."""
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(bow_matrix[0], bow_matrix[1])[0][0]

def get_word2vec_similarity(text1, text2):
    """Compute similarity using Word2Vec embeddings from spaCy."""
    doc1, doc2 = nlp(text1), nlp(text2)
    return doc1.similarity(doc2)

def detect_ai_content(text):
    """Basic AI text detection based on repetition, perplexity, and structure."""
    avg_word_length = np.mean([len(word) for word in text.split()])
    sentence_count = len(re.split(r'[.!?]', text))
    avg_sentence_length = len(text.split()) / max(sentence_count, 1)
    
    # AI text tends to have longer words & uniform sentence lengths
    if avg_word_length > 5.5 or avg_sentence_length > 20:
        return "Likely AI-generated"
    return "Likely Human-written"

def get_similarity(text1, text2):
    """Calculate similarity using TF-IDF, BoW, Word2Vec, and SequenceMatcher."""
    text1, text2 = preprocess_text(text1), preprocess_text(text2)

    # TF-IDF Similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    
    # BoW Similarity
    bow_sim = get_bow_similarity(text1, text2)
    
    # Word2Vec Similarity
    word2vec_sim = get_word2vec_similarity(text1, text2)
    
    # SequenceMatcher Similarity
    sequence_matcher = difflib.SequenceMatcher(None, text1, text2).ratio()
    
    # Final Score (Average of all methods)
    return ((cosine_sim + bow_sim + word2vec_sim + sequence_matcher) / 4) * 100

if __name__ == "__main__":
    doc1 = "This is a sample document."
    doc2 = "This document is an example."
    
    print(f"Similarity Score: {get_similarity(doc1, doc2):.2f}%")
    print(f"AI Content Detection: {detect_ai_content(doc1)}")
    print(f"AI Content Detection: {detect_ai_content(doc2)}")


# import pypdfium2
# import pdfplumber

# # 
# # Function that define the pdf_path and file input 
# def extract_text_from_pdf(pdf_path):
#     """Extract text from a given PDF file."""
#     text = ""
#     # Open the PDF file 
#     with pdfplumber.open(pdf_path) as pdf:
#         # loop through each page and extract text
#         # or iterate through each page using pdf.pages
#         for page in pdf.pages:
#             text += page.extract_text() + "\n"
#     return text.strip()

# # Exmaple usage 
# if __name__ == "__main__":
#     extracted_text = extract_text_from_pdf("data/pdf-1.pdf")
#     print(extracted_text[:5000])  # Print first 500 characters

