import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# FIX 1: Use the MULTILINGUAL version of the model
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

def clean_text(text):
    # Remove YouTube timestamps (e.g., 10:45) which confuse the model
    text = re.sub(r'\d+:\d+', '', text)
    # Remove excessive whitespace
    text = " ".join(text.split())
    return text

def predict_sentiment(texts, batch_size=16):
    tokenizer, model = load_model()
    
    # The multilingual model uses: 0: Negative, 1: Neutral, 2: Positive
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    # Fix 2: Keywords that AI often misinterprets as 'Negative' in Tamil/English
    # These are usually requests for more content.
    neutral_keywords = ['podunga', 'pannunga', 'sollunga', 'venum', 'review please', 'suggest', 'comparison', 'waiting']
    positive_keywords = ['super', 'nice', 'mass', 'fire', '🔥', '👍', '❤️', 'love', 'thala']

    all_preds = []
    all_scores = []
    model.eval()

    # Pre-clean the texts
    cleaned_texts = [clean_text(str(t)) for t in texts]

    for i in range(0, len(cleaned_texts), batch_size):
        batch = cleaned_texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        for j, p in enumerate(preds):
            label = label_map.get(p.item(), "Neutral")
            score = torch.max(probs[j]).item()
            original_text = batch[j].lower()

            # FIX 3: Rule-based refinement for common YouTube errors
            # If the model thinks it's Negative, but it contains request keywords, move to Neutral
            if label == "Negative":
                if any(word in original_text for word in neutral_keywords):
                    label = "Neutral"
                elif any(word in original_text for word in positive_keywords):
                    label = "Positive"
            
            # If the model thinks it's Neutral, but has strong positive words/emojis
            if label == "Neutral" and any(word in original_text for word in positive_keywords):
                label = "Positive"

            all_preds.append(label)
            all_scores.append(round(score, 4))

    return all_preds, all_scores
