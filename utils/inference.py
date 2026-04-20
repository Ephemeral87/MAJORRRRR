import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "your-username/indicbert-youtube-sentiment"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model


def predict_sentiment(texts, batch_size=16):
    tokenizer, model = load_model()
    label_map = {0: "Positive", 1: "Neutral", 2: "Negative", 3: "Other"}

    all_preds = []
    all_scores = []

    model.eval()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        all_preds.extend([label_map[p.item()] for p in preds])
        all_scores.extend([round(torch.max(prob).item(), 4) for prob in probs])

    return all_preds, all_scores
