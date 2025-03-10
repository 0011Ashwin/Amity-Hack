# difflib is used for comparing and matching the sequences
import difflib
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the small English model from spaCy and also its a part of nlp 
nlp = spacy.load("en_core_web_sm")

# Text pre-processing fucntion for text what we provide
# Handling the lowercase of text then applying lemmatization and removing stopwords 
# Then return the text as a string 
def preprocess_text(text):
    """Preprocess text: lowercasing, lemmatization, and removing stopwords."""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop])


# Converts the cleaned texts into numerical representations.
# TF-IDF -> Term Frequency-Inverse Document Frequency
def get_similarity(text1, text2):
    """Calculate similarity using TF-IDF and SequenceMatcher."""
    text1, text2 = preprocess_text(text1), preprocess_text(text2)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    # Measures how similar the documents are based on word distribution.
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    sequence_matcher = difflib.SequenceMatcher(None, text1, text2).ratio()
    
    # Multiple by 100 to get percentage.
    return ((cosine_sim + sequence_matcher) / 2) * 100

if __name__ == "__main__":
    doc1 = "This is a sample document."
    doc2 = "This document is an example."
    print(f"Similarity Score: {get_similarity(doc1, doc2):.2f}%")

