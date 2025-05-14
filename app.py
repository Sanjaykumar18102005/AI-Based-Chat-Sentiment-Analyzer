import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.eval()

# Define labels
labels = ['Negative', 'Neutral', 'Positive']

# Streamlit app
st.set_page_config(page_title="AI-Based Chat Sentiment Analyzer", layout="centered")

st.title("🧠 AI-Based Chat Sentiment Analyzer")
st.write("Type a chat message below to analyze its sentiment using a BERT model.")

# Text input
user_input = st.text_area("Enter your chat message:", height=100)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        # Tokenize and predict
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(probs, dim=1).item()
            confidence = np.max(probs.numpy()) * 100

        # Output
        st.success(f"**Sentiment:** {labels[predicted_label]}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        # Optional emoji feedback
        emojis = ["😠", "😐", "😊"]
        st.markdown(f"### {emojis[predicted_label]}")
