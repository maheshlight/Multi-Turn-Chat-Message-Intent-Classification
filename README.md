# 💬 Multi-Turn Chat Message Intent Classification & Comment Analyzer

> **Team 1 | Module 7 | NLP Final Project | Python & Jupyter Notebook**

---

## 📌 Project Overview

This project builds an end-to-end **NLP pipeline for Multi-Turn Chat Message Intent Classification**. It classifies chat messages, customer reviews, and multi-turn dialogues into **9 intent categories** using both traditional ML models and a **BERT deep learning classifier**, enhanced with a live **Claude AI-powered Comment Analyzer**.

**Goal:** Classify chat messages into intents — Query, Complaint, Greeting, Farewell, Request, Appreciation, Escalation, Feedback, and Clarification.

---

## 👥 Team Members

- Ankit
- Samir
- Sushant
- Mahesh
- Pooja

---

## 🛠️ Technologies Used

| Category | Tools / Libraries |
|----------|------------------|
| **Language** | Python 3 |
| **Notebook Environment** | Jupyter Notebook |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **NLP Preprocessing** | SpaCy (`en_core_web_sm`) |
| **Feature Extraction** | TF-IDF Vectorizer (Scikit-learn) |
| **ML Models** | Logistic Regression, LinearSVC, Multinomial Naive Bayes, Random Forest, Gradient Boosting |
| **Deep Learning Model** | BERT (`BertForSequenceClassification`, `BertTokenizer`) |
| **Deep Learning Framework** | PyTorch, Hugging Face Transformers |
| **ML Utilities** | Scikit-learn Pipeline, LabelEncoder, train_test_split, AdamW Optimizer |
| **ML Metrics** | Accuracy Score, Classification Report, Confusion Matrix |
| **AI Comment Analyzer** | Anthropic Claude API (`claude-sonnet-4-20250514`) |
| **Scheduler** | Hugging Face `get_linear_schedule_with_warmup` |

---

## 🗂️ Dataset

| Field | Detail |
|-------|--------|
| **Type** | Custom Large-Scale Intent Dataset (built in-notebook) |
| **Intent Categories** | 9 |
| **Examples per Category** | 100+ |
| **Total Intents** | Query, Complaint, Greeting, Farewell, Request, Appreciation, Escalation, Feedback, Clarification |

---

## 🧠 Intent Categories

| # | Intent | Description |
|---|--------|-------------|
| 1 | **Query** | User asking for information |
| 2 | **Complaint** | User expressing dissatisfaction |
| 3 | **Greeting** | User initiating or opening a conversation |
| 4 | **Farewell** | User closing the conversation |
| 5 | **Request** | User asking to perform an action |
| 6 | **Appreciation** | User expressing thanks or positive feedback |
| 7 | **Escalation** | User demanding urgent attention or higher support |
| 8 | **Feedback** | User providing suggestions or opinions |
| 9 | **Clarification** | User seeking to clear confusion or misunderstanding |

---

## 📋 Project Steps

| Step | Description |
|------|-------------|
| **Step 0** | Interactive Multi-Turn Conversation Intent Analyzer (Claude AI powered) |
| **Step 1** | Install & Import Libraries |
| **Step 2** | Build Large-Scale Intent Dataset (9 categories, 100+ examples each) |
| **Step 3** | Exploratory Data Analysis (EDA) |
| **Step 4** | Text Preprocessing with SpaCy |
| **Step 5** | Encode Labels & Split Dataset |
| **Step 6** | ML Models with TF-IDF Features |
| **Step 7** | Detailed Classification Report (Best ML Model) |
| **Step 8** | Model Comparison Chart |
| **Step 9** | BERT-Based Deep Learning Classifier |
| **Step 10** | BERT Training Curves & Test Evaluation |
| **Step 11** | Final Model Comparison Summary |
| **Step 12** | Live Intent Prediction Function |
| **Step 13** | Multi-Turn Context Analysis |
| **Step 14** | Save Best Model |
| **Step 15** | Paste-a-Comment / Review Analyzer |

---

## ✨ Key Features

**Two Analysis Modes:**
- **Option A — Single Text:** Paste any comment or review for instant intent analysis
- **Option B — Multi-Turn Chat:** Analyze an entire conversation turn-by-turn with intent flow tracking

**Claude AI Analyzer outputs:**
- Primary Intent & Secondary Intents
- Emotional Tone (Neutral / Positive / Negative / Mixed)
- Urgency Level (Low / Medium / High)
- Key Intent Signals (words/phrases driving classification)
- Escalation Detection
- Recommended Action

---

## 🚀 How to Run

1. Make sure **Python 3** and **pip** are installed.
2. Install required libraries:
   ```bash
   pip install transformers torch scikit-learn spacy pandas numpy matplotlib seaborn datasets accelerate anthropic
   python -m spacy download en_core_web_sm
   ```
3. Set your **Anthropic API key** (for Step 0 & Step 15):
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```
4. Open the notebook:
   ```bash
   jupyter notebook Multi_Turn_Chat_Intent_Classification_with_Comment_Analyzer_FINAL_PROJECT_TEAM_1.ipynb
   ```
5. Run all cells from top to bottom (`Kernel → Restart & Run All`).

---

## 📊 Key Outputs

- Intent distribution charts (EDA)
- ML model accuracy comparison (Logistic Regression, SVM, Naive Bayes, Random Forest, Gradient Boosting)
- Confusion matrix for best ML model
- BERT training & validation loss/accuracy curves
- Final side-by-side model comparison (ML vs BERT)
- Live real-time intent prediction on custom input
- Multi-turn intent flow with escalation detection
- Saved best model for deployment

---

## 📅 Project Info

| Field | Detail |
|-------|--------|
| **Project Type** | NLP Final Project |
| **Module** | Module 7 |
| **Language** | Python 3 |
| **Notebook Format** | `.ipynb` (Jupyter Notebook) |
| **Domain** | Natural Language Processing / Conversational AI |
| **Models Used** | 5 ML Models + BERT + Claude AI |

---

## 📝 License

This project is created for **educational purposes only**. All data used is synthetically generated for academic demonstration.
