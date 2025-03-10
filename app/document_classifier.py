# import torch
# from transformers import BertTokenizer, BertForSequenceClassification

# # Model name
# MODEL_NAME = "bhadresh-savani/bert-base-uncased-emotion"

# # Load tokenizer and model
# tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
# model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

# # Define category labels
# LABELS = ["Financial", "Healthcare", "Legal"]

# def classify_document(text: str) -> str:
#     """Classifies a document into Financial, Healthcare, or Legal."""
#     if not text.strip():  # Handle empty input
#         return "Invalid input: Text is empty"
    
#     # Tokenize and prepare inputs
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
#     # Get model predictions
#     with torch.no_grad():
#         logits = model(**inputs).logits
    
#     # Get the predicted category
#     predicted_label = torch.argmax(logits, dim=1).item()
    
#     return LABELS[predicted_label]

# # Example usage
# if __name__ == "__main__":
#     sample_text = "The stock market reported a significant increase in quarterly revenue."
#     print("Predicted Category:", classify_document(sample_text))
