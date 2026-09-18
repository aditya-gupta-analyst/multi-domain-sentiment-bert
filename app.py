 import streamlit as st
import torch
from transformers import BertForSequenceClassification, AutoTokenizer

st.set_page_config(page_title="FinCard AI Sentiment Dashboard", layout="wide")

MODEL_REPO = "adityagupta-data/multi-domain-sentiment-bert"

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(MODEL_REPO).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    return model, tokenizer, device

st.title("📊 Multi-Domain Sentiment Analysis using BERT")
st.markdown("**Transfer Learning Evaluation across Movie Reviews, Product Reviews, and Tweets**")

st.header("Model Performance Comparison")
st.table({
    "Model": ["Naive Bayes", "Logistic Regression", "SVM (Linear)", "BERT (Transfer Learning)"],
    "Accuracy": ["75.02%", "79.03%", "78.59%", "86.22%"],
    "F1-Score": ["75.46%", "79.14%", "78.55%", "86.26%"]
})

st.header("Cross-Domain BERT Performance")
st.table({
    "Domain": ["IMDB (Movies)", "Amazon (Products)", "Tweets"],
    "Accuracy": ["92.73%", "93.10%", "71.10%"],
    "F1-Score": ["92.73%", "93.19%", "71.01%"]
})

st.header("Try a Live Prediction")
user_text = st.text_area("Enter text to analyze sentiment:", "This movie was absolutely fantastic!")
if st.button("Predict"):
    with st.spinner("Loading model and predicting..."):
        model, tokenizer, device = load_model()
        encoding = tokenizer(user_text, truncation=True, padding="max_length", max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoding)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred = torch.argmax(probs).item()
        labels_map = {0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😊"}
        st.success(f"**Prediction: {labels_map[pred]}**")
        st.write(f"Confidence — Negative: {probs[0]:.2%} | Neutral: {probs[1]:.2%} | Positive: {probs[2]:.2%}")

st.header("Conclusion")
st.markdown("""
- **Key Finding:** BERT (86.2% accuracy) outperforms all classical ML models.
- **Challenge:** The Neutral class, present only in Tweets, was the primary source of misclassification.
- **Application:** Powers the sentiment-reading component of the FinCard AI project.
""")
