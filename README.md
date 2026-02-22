# 🕵️ AI Forensic Auditor

A Machine Learning application that detects AI-generated text by analyzing "Linguistic DNA"—specifically **Perplexity** and **Burstiness**.

![Project Screenshot](Screenshot.png)

## 💡 How it Works
This project doesn't just "guess." It extracts mathematical features from text:
- **Perplexity:** Uses GPT-2 to measure how "predictable" the word choices are.
- **Burstiness:** Measures the variation in sentence length (Human writing has "bursts").
- **Classification:** A Random Forest model trained on 500k+ samples to provide a final probability score.

## 🛠️ Tech Stack
- **Python** (Data Science)
- **Scikit-Learn** (Random Forest Classifier)
- **HuggingFace Transformers** (GPT-2 for feature extraction)
- **Streamlit/Gradio** (Web UI)

## 🚀 Run it Locally
1. Clone this repo:
   git clone https://github.com/Sahithi987654/AI-Forensic-Auditor.git
3. Install dependencies: `pip install -r requirements.txt`
4. Launch the app: `streamlit run app.py`
