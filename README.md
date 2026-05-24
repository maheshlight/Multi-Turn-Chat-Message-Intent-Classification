# 🧠 Multi-Turn Chat Intent Classifier
**Team 1 · Module 7 NLP Project**

Classify chat messages and multi-turn conversations into 9 intent categories using **Claude AI (Anthropic)** and trained **ML models (TF-IDF + Scikit-learn / BERT)**.

---

## 🎯 Intent Categories
| Intent | Description |
|---|---|
| Query | User asking for information |
| Complaint | User expressing dissatisfaction |
| Greeting | User opening a conversation |
| Farewell | User closing a conversation |
| Request | User asking to perform an action |
| Appreciation | User expressing thanks |
| Escalation | User demanding higher authority |
| Feedback | User providing suggestions |
| Clarification | User asking to clarify something |

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/your-username/intent-classifier.git
cd intent-classifier
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Set your API key
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```
Or set it as an environment variable:
```bash
export ANTHROPIC_API_KEY=your_key_here
```

### 3. (Optional) Train ML models
Run the notebook `Multi_Turn_Chat_Intent_Classification...ipynb` to train and save models to `saved_models/`.

### 4. Run the Streamlit app
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure
```
intent-classifier/
├── app.py                          # Streamlit web app
├── requirements.txt                # Python dependencies
├── .env.example                    # API key template
├── README.md                       # This file
├── Multi_Turn_Chat_Intent_...ipynb # Training notebook
└── saved_models/                   # Created after training
    ├── best_ml_model_*.pkl
    ├── tfidf_vectorizer.pkl
    ├── label_encoder.pkl
    └── bert_intent_classifier/
```

---

## 🌐 Deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `app.py`
4. Under **Settings → Secrets**, add:
   ```
   ANTHROPIC_API_KEY = "your_key_here"
   ```
5. Click **Deploy** 🎉

---

## 🛠️ Tech Stack
- **Claude API** (Anthropic) — deep intent analysis
- **Streamlit** — web UI
- **Scikit-learn** — TF-IDF + ML classifiers
- **BERT** (HuggingFace Transformers) — deep learning classifier
- **SpaCy** — text preprocessing
