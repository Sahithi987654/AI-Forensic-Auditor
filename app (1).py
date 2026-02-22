import streamlit as st
import joblib
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import re
import numpy as np

# Page Config
st.set_page_config(page_title="AI Forensic Auditor", page_icon="🕵️")
st.title("🕵️ AI Forensic Auditor")
st.markdown("Analyze text for 'Linguistic DNA' to detect AI generation.")

# Load Model & Detector Logic
@st.cache_resource
def load_tools():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    clf = joblib.load('ai_detector_model.pkl')
    return model, tokenizer, clf

model, tokenizer, clf = load_tools()

def calculate_forensics(text):
    sentences = re.split(r'\.|\!|\?', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
    burstiness = np.std([len(s.split()) for s in sentences]) if len(sentences) > 1 else 0.5

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        loss = model(inputs.input_ids, labels=inputs.input_ids).loss
        perplexity = torch.exp(loss).item()
    return perplexity, burstiness

# User Input
user_text = st.text_area("Paste text here (at least 20 words):", height=200)

if st.button("Analyze Text"):
    if len(user_text.split()) < 20:
        st.warning("Please enter more text for a reliable audit.")
    else:
        with st.spinner("Analyzing Linguistic Fingerprints..."):
            p, b = calculate_forensics(user_text)
            prob = clf.predict_proba([[p, b]])[0][1]

            # Calibration adjustment
            if b > 12: prob -= 0.2
            prob = max(0, min(1, prob))

            # Display Results
            st.subheader(f"AI Probability: {prob*100:.1f}%")
            st.progress(prob)

            if prob > 0.6:
                st.error("🚩 Verdict: Likely AI-Generated")
            else:
                st.success("🟢 Verdict: Likely Human-Written")

            col1, col2 = st.columns(2)
            col1.metric("Perplexity", f"{p:.2f}")
            col2.metric("Burstiness", f"{b:.2f}")
