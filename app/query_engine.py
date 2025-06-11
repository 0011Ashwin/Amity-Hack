def query_pdf(text, question):
    """
    Query PDF content with a question.
    This is a wrapper around query_groq for backward compatibility.
    """
    from app.pdf_extractor import query_groq
    return query_groq(text, question)
