import difflib
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the small English model from spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If model not found, download it first
    import subprocess
    import sys
    
    st.warning("Downloading language model for first-time use...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    st.error(f"Error loading spaCy model: {str(e)}")
    # Create a minimal fallback
    import en_core_web_sm
    nlp = en_core_web_sm.load()

def preprocess_text(text):
    """Preprocess text: lowercasing, lemmatization, and removing stopwords."""
    try:
        if not text or not isinstance(text, str):
            return ""
        
        # Process smaller chunks if text is large
        if len(text) > 100000:
            # Process first 100k characters to avoid memory issues
            text = text[:100000]
            st.warning("Text was truncated to 100,000 characters for processing.")
            
        doc = nlp(text.lower())
        return " ".join([token.lemma_ for token in doc if not token.is_stop])
    except Exception as e:
        st.error(f"Error in text preprocessing: {str(e)}")
        # Fallback to simple processing
        return text.lower() if isinstance(text, str) else ""

def get_similarity(text1, text2):
    """Calculate similarity using TF-IDF and SequenceMatcher."""
    try:
        # Handle empty inputs
        if not text1 or not text2:
            return 0.0
        
        # Preprocess texts
        try:
            text1_processed = preprocess_text(text1)
            text2_processed = preprocess_text(text2)
            
            if not text1_processed or not text2_processed:
                return 0.0
        except Exception as e:
            st.error(f"Error preprocessing text: {str(e)}")
            # Fallback to simple lowercase
            text1_processed = text1.lower() if isinstance(text1, str) else ""
            text2_processed = text2.lower() if isinstance(text2, str) else ""

        # TF-IDF Vectorization
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1_processed, text2_processed])
            cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        except Exception as e:
            st.error(f"Error in TF-IDF calculation: {str(e)}")
            cosine_sim = 0.0
        
        # Sequence matching
        try:
            sequence_matcher = difflib.SequenceMatcher(None, text1_processed, text2_processed).ratio()
        except Exception as e:
            st.error(f"Error in sequence matching: {str(e)}")
            sequence_matcher = 0.0
        
        # Average the two similarity measures and convert to percentage
        similarity = ((cosine_sim + sequence_matcher) / 2) * 100
        
        return float(similarity)
    except Exception as e:
        st.error(f"General error in similarity calculation: {str(e)}")
        return 0.0

def detailed_similarity_analysis(text1, text2):
    """Provides a more detailed analysis of similarity between texts."""
    try:
        # Basic similarity
        overall_similarity = get_similarity(text1, text2)
        
        # Get matching sequences
        text1_words = text1.lower().split()
        text2_words = text2.lower().split()
        
        # Find common words
        common_words = set(text1_words).intersection(set(text2_words))
        
        # Find significant common sequences (3+ words in a row)
        sequences = []
        min_sequence_length = 3
        
        for i in range(len(text1_words) - min_sequence_length + 1):
            sequence = text1_words[i:i+min_sequence_length]
            seq_str = " ".join(sequence)
            
            if seq_str in " ".join(text2_words):
                sequences.append(seq_str)
                
        # Remove duplicates
        significant_sequences = list(set(sequences))
        
        # Limit to top 10 sequences
        top_sequences = significant_sequences[:10]
        
        return {
            "overall_similarity": overall_similarity,
            "common_word_count": len(common_words),
            "significant_words": sorted(list(common_words), key=len, reverse=True)[:20],
            "matching_sequences": top_sequences
        }
    except Exception as e:
        st.error(f"Error in detailed analysis: {str(e)}")
        return {
            "overall_similarity": 0.0,
            "common_word_count": 0,
            "significant_words": [],
            "matching_sequences": []
        }

# Test function if run directly
if __name__ == "__main__":
    doc1 = "This is a sample document with some specific terminology and phrasing."
    doc2 = "This document is an example with similar specific terminology."
    print(f"Similarity Score: {get_similarity(doc1, doc2):.2f}%")
    print("Detailed Analysis:", detailed_similarity_analysis(doc1, doc2))
