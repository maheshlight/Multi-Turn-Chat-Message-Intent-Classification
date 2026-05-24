"""
app.py — Multi-Turn Chat Intent Classification
Team 1 | Module 7 NLP Project
Run:  streamlit run app.py
"""

import os
import re
import json
import joblib
import numpy as np
import streamlit as st
import google.generativeai as genai

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intent Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
INTENT_LABELS = [
    "Query", "Complaint", "Greeting", "Farewell", "Request",
    "Appreciation", "Escalation", "Feedback", "Clarification"
]

INTENT_COLORS = {
    "Query":         "#3B82F6",
    "Complaint":     "#EF4444",
    "Greeting":      "#10B981",
    "Farewell":      "#6B7280",
    "Request":       "#8B5CF6",
    "Appreciation":  "#F59E0B",
    "Escalation":    "#DC2626",
    "Feedback":      "#06B6D4",
    "Clarification": "#F97316",
}

INTENT_EMOJI = {
    "Query":         "❓",
    "Complaint":     "😤",
    "Greeting":      "👋",
    "Farewell":      "👋",
    "Request":       "📋",
    "Appreciation":  "🌟",
    "Escalation":    "🚨",
    "Feedback":      "💡",
    "Clarification": "🔍",
}

SAVED_MODELS_DIR = "saved_models"

# ─────────────────────────────────────────────────────────────────────────────
# Load ML models (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_ml_models():
    """Load TF-IDF vectorizer, best ML model, and label encoder if available."""
    try:
        import glob
        pkl_files = glob.glob(os.path.join(SAVED_MODELS_DIR, "best_ml_model_*.pkl"))
        if not pkl_files:
            return None, None, None
        ml_model  = joblib.load(pkl_files[0])
        tfidf     = joblib.load(os.path.join(SAVED_MODELS_DIR, "tfidf_vectorizer.pkl"))
        le        = joblib.load(os.path.join(SAVED_MODELS_DIR, "label_encoder.pkl"))
        return ml_model, tfidf, le
    except Exception:
        return None, None, None

ml_model, tfidf, le = load_ml_models()
ML_AVAILABLE = ml_model is not None

# ─────────────────────────────────────────────────────────────────────────────
# SpaCy preprocessing (optional, graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None

nlp_spacy = load_spacy()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    if nlp_spacy:
        doc = nlp_spacy(text)
        tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and len(t.lemma_) > 2]
        return " ".join(tokens)
    # simple fallback
    return " ".join(w for w in text.split() if len(w) > 2)

# ─────────────────────────────────────────────────────────────────────────────
# Gemini client (Google AI — free tier)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_gemini_model():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")

gemini_model = get_gemini_model()

# ─────────────────────────────────────────────────────────────────────────────
# Tone / urgency helpers
# ─────────────────────────────────────────────────────────────────────────────
_NEG = ["frustrated","angry","terrible","worst","awful","unacceptable","useless",
        "rude","damaged","wrong","broken","horrible","fed up","disappointed",
        "furious","upset","disgusted","ridiculous","not working"]
_POS = ["thank","thanks","appreciate","great","amazing","excellent","happy",
        "love","fantastic","brilliant","perfect","wonderful","helpful",
        "pleased","satisfied","impressed","recommend"]
_URG_HIGH = ["immediately","urgent","asap","right now","emergency","now",
             "critical","right away","as soon as possible"]
_URG_MED  = ["soon","today","quickly","need this","please help","still waiting"]

def emotional_tone(text: str) -> str:
    t = text.lower()
    neg = sum(1 for k in _NEG if k in t)
    pos = sum(1 for k in _POS if k in t)
    if neg > 0 and pos > 0: return "😐 Mixed"
    if neg > 0:              return "😠 Negative"
    if pos > 0:              return "😊 Positive"
    return "😶 Neutral"

def urgency_level(text: str) -> str:
    t = text.lower()
    if any(k in t for k in _URG_HIGH): return "🔴 High"
    if any(k in t for k in _URG_MED):  return "🟡 Medium"
    return "🟢 Low"

def key_signals(text: str) -> list:
    all_kw = _NEG + _POS + _URG_HIGH + _URG_MED
    t = text.lower()
    found = list(dict.fromkeys(k for k in all_kw if k in t))
    return found[:6] or ["(no strong signals)"]

# ─────────────────────────────────────────────────────────────────────────────
# ML prediction
# ─────────────────────────────────────────────────────────────────────────────
def predict_ml(text: str):
    if not ML_AVAILABLE:
        return None, None
    processed = preprocess_text(text)
    vec  = tfidf.transform([processed])
    pred = ml_model.predict(vec)[0]
    label = le.inverse_transform([pred])[0]
    conf = None
    if hasattr(ml_model, "predict_proba"):
        probs = ml_model.predict_proba(vec)[0]
        conf  = round(float(probs.max()) * 100, 1)
    return label.capitalize(), conf

# ─────────────────────────────────────────────────────────────────────────────
# Gemini prediction — single text
# ─────────────────────────────────────────────────────────────────────────────
def analyze_with_claude_single(text: str) -> dict:
    prompt = f"""You are an expert NLP intent classification system.
Analyze text and identify conversation intents.
Always respond in valid JSON only — no preamble, no markdown fences.

Analyze the following text for multi-turn conversation intent.

TEXT:
\"\"\"{text.strip()}\"\"\"

Identify:
1. Primary intent (choose from: Query, Complaint, Greeting, Farewell, Request, Appreciation, Escalation, Feedback, Clarification)
2. Secondary intents if present
3. Emotional tone (Neutral / Positive / Negative / Mixed)
4. Urgency level (Low / Medium / High)
5. Key intent signals (words/phrases driving the classification)
6. A short explanation of the classification

Return ONLY a JSON object with this exact structure:
{{
  "primary_intent": "...",
  "secondary_intents": ["...", "..."],
  "emotional_tone": "...",
  "urgency": "...",
  "key_signals": ["...", "..."],
  "explanation": "..."
}}"""

    response = gemini_model.generate_content(prompt)
    raw = response.text.strip()
    raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)

# ─────────────────────────────────────────────────────────────────────────────
# Gemini prediction — multi-turn
# ─────────────────────────────────────────────────────────────────────────────
def analyze_with_claude_multiturn(turns: list) -> dict:
    conv_text = "\n".join(
        f"Turn {t['turn']} ({t['speaker']}): {t['message']}" for t in turns
    )
    prompt = f"""You are an expert NLP intent classification system specializing in multi-turn dialogue.
Analyze each turn and the overall conversation flow.
Always respond in valid JSON only — no preamble, no markdown fences.

Analyze the following multi-turn conversation for intent classification.

CONVERSATION:
{conv_text}

For each turn identify its intent. Then provide an overall analysis.

Return ONLY a JSON object with this exact structure:
{{
  "turns": [
    {{
      "turn": 1,
      "message_preview": "...",
      "intent": "...",
      "confidence": "High/Medium/Low",
      "emotional_tone": "Neutral/Positive/Negative/Mixed"
    }}
  ],
  "intent_flow": ["intent1", "intent2"],
  "dominant_intent": "...",
  "conversation_summary": "...",
  "escalation_detected": true,
  "overall_tone": "...",
  "recommended_action": "..."
}}"""

    response = gemini_model.generate_content(prompt)
    raw = response.text.strip()
    raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)

# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def intent_badge(intent: str) -> str:
    color = INTENT_COLORS.get(intent, "#6B7280")
    emoji = INTENT_EMOJI.get(intent, "🔵")
    return f"""<span style="background:{color};color:#fff;padding:3px 10px;border-radius:12px;
                font-weight:600;font-size:0.85rem;">{emoji} {intent}</span>"""

def confidence_bar(conf):
    if conf is None:
        return ""
    color = "#10B981" if conf >= 75 else "#F59E0B" if conf >= 50 else "#EF4444"
    return f"""
    <div style="background:#e5e7eb;border-radius:6px;height:10px;width:100%;">
      <div style="background:{color};border-radius:6px;height:10px;width:{conf}%;"></div>
    </div>
    <small style="color:{color};font-weight:600;">{conf}%</small>"""

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Intent Classifier")
    st.caption("Team 1 · Module 7 NLP")
    st.divider()

    mode = st.radio(
        "Analysis Mode",
        ["Single Text / Comment", "Multi-Turn Conversation"],
        index=0,
    )
    st.divider()

    engine = st.radio(
        "Classification Engine",
        ["Gemini AI (Google)" if True else "Gemini AI",
         "ML Model (TF-IDF)" if ML_AVAILABLE else "ML Model (not loaded)"],
        index=0,
    )
    use_claude = "Gemini" in engine

    st.divider()
    st.markdown("**Intent Categories**")
    for label in INTENT_LABELS:
        color = INTENT_COLORS[label]
        emoji = INTENT_EMOJI[label]
        st.markdown(
            f"<span style='color:{color};font-weight:600;'>{emoji} {label}</span>",
            unsafe_allow_html=True,
        )

    st.divider()
    if not ML_AVAILABLE:
        st.warning("⚠️ ML model not found.\nTrain first via the notebook and place outputs in `saved_models/`.")
    else:
        st.success("✅ ML model loaded")

# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────
st.title("💬 Multi-Turn Chat Intent Classifier")
st.caption("Classify chat messages · Powered by Claude AI + ML Models")
st.divider()

# ─── Single Text Mode ────────────────────────────────────────────────────────
if mode == "Single Text / Comment":
    st.subheader("📝 Single Comment / Review Analyzer")

    example_texts = {
        "Select an example…": "",
        "Complaint":       "I ordered a laptop last week and it still hasn't arrived. The tracking page just says 'in transit' for 5 days. Nobody from support is responding to my emails.",
        "Appreciation":    "Thank you so much for your help! You've been amazing and sorted everything so quickly!",
        "Escalation":      "I want to speak to your manager right now. This needs to be escalated immediately.",
        "Query":           "What is your return policy for electronics? Do you offer free returns?",
        "Request":         "Please cancel my order and issue a full refund to my original payment method.",
        "Feedback":        "Your app is great but could be more user-friendly. I'd suggest adding a dark mode option.",
        "Greeting":        "Hello! I'm a new customer and need some assistance please.",
        "Farewell":        "Thanks for all your help today. Have a great day! Goodbye!",
        "Clarification":   "I didn't understand what you meant by 'processing time'. Can you clarify?",
    }

    selected = st.selectbox("Try an example:", list(example_texts.keys()))
    default_text = example_texts[selected]

    user_text = st.text_area(
        "Paste your comment, review, or chat message:",
        value=default_text,
        height=150,
        placeholder="Type or paste any text here…",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear"):
            user_text = ""

    if analyze_btn and user_text.strip():
        with st.spinner("Analyzing intent…"):
            # ── Local tone/urgency (always shown)
            tone    = emotional_tone(user_text)
            urgency = urgency_level(user_text)
            signals = key_signals(user_text)

            cols = st.columns(2)

            # ── Gemini result
            with cols[0]:
                st.markdown("#### 🤖 Gemini AI Analysis")
                try:
                    result = analyze_with_claude_single(user_text)
                    primary = result.get("primary_intent", "Unknown")
                    secondary = result.get("secondary_intents", [])
                    c_tone = result.get("emotional_tone", tone)
                    c_urgency = result.get("urgency", "N/A")
                    c_signals = result.get("key_signals", signals)
                    explanation = result.get("explanation", "")

                    st.markdown(f"**Primary Intent:** {intent_badge(primary)}", unsafe_allow_html=True)
                    if secondary:
                        badges = " ".join(intent_badge(s) for s in secondary)
                        st.markdown(f"**Secondary:** {badges}", unsafe_allow_html=True)
                    st.markdown(f"**Tone:** {c_tone}")
                    st.markdown(f"**Urgency:** {c_urgency}")
                    st.markdown(f"**Key Signals:** `{'`, `'.join(c_signals)}`")
                    if explanation:
                        st.info(f"💬 {explanation}")
                except Exception as e:
                    st.error(f"Gemini API error: {e}")

            # ── ML result
            with cols[1]:
                st.markdown("#### 🔬 ML Model Analysis")
                if ML_AVAILABLE:
                    ml_label, ml_conf = predict_ml(user_text)
                    st.markdown(f"**Predicted Intent:** {intent_badge(ml_label)}", unsafe_allow_html=True)
                    if ml_conf is not None:
                        st.markdown("**Confidence:**", unsafe_allow_html=True)
                        st.markdown(confidence_bar(ml_conf), unsafe_allow_html=True)
                    st.markdown(f"**Tone (local):** {tone}")
                    st.markdown(f"**Urgency (local):** {urgency}")
                    st.markdown(f"**Key Signals:** `{'`, `'.join(signals)}`")
                else:
                    st.warning("ML model not loaded. Train the model first.")
                    st.markdown(f"**Tone (local):** {tone}")
                    st.markdown(f"**Urgency (local):** {urgency}")
                    st.markdown(f"**Key Signals:** `{'`, `'.join(signals)}`")

    elif analyze_btn:
        st.warning("Please enter some text to analyze.")

# ─── Multi-Turn Mode ─────────────────────────────────────────────────────────
else:
    st.subheader("🔄 Multi-Turn Conversation Analyzer")

    # Session state for conversation turns
    if "turns" not in st.session_state:
        st.session_state.turns = []
    if "turn_counter" not in st.session_state:
        st.session_state.turn_counter = 1

    # Pre-built example
    EXAMPLE_CONV = [
        {"turn": 1, "speaker": "User", "message": "Hi there! Good morning!"},
        {"turn": 2, "speaker": "User", "message": "What are the available plans for new customers?"},
        {"turn": 3, "speaker": "User", "message": "I don't understand what the 'pro tier' includes."},
        {"turn": 4, "speaker": "User", "message": "Please sign me up for the premium plan."},
        {"turn": 5, "speaker": "User", "message": "Wait, I was charged double! This is unacceptable!"},
        {"turn": 6, "speaker": "User", "message": "Get me your manager now. This needs to be escalated."},
        {"turn": 7, "speaker": "User", "message": "I think you should add better billing transparency."},
        {"turn": 8, "speaker": "User", "message": "Thanks for sorting it out. Really appreciate it!"},
        {"turn": 9, "speaker": "User", "message": "That's all I needed. Take care, goodbye!"},
    ]

    col_add, col_example, col_clear = st.columns([2, 2, 1])
    with col_example:
        if st.button("📂 Load Example Conversation"):
            st.session_state.turns = EXAMPLE_CONV.copy()
            st.session_state.turn_counter = len(EXAMPLE_CONV) + 1
    with col_clear:
        if st.button("🗑️ Clear All"):
            st.session_state.turns = []
            st.session_state.turn_counter = 1

    # Add new turn
    with st.expander("➕ Add a Turn", expanded=not st.session_state.turns):
        speaker = st.selectbox("Speaker", ["User", "Agent", "System"])
        new_msg = st.text_input("Message", placeholder="Type a message…")
        if st.button("Add Turn"):
            if new_msg.strip():
                st.session_state.turns.append({
                    "turn":    st.session_state.turn_counter,
                    "speaker": speaker,
                    "message": new_msg.strip(),
                })
                st.session_state.turn_counter += 1
                st.rerun()

    # Show current turns
    if st.session_state.turns:
        st.markdown("#### 💬 Conversation")
        for t in st.session_state.turns:
            color = "#3B82F6" if t["speaker"] == "User" else "#10B981"
            st.markdown(
                f"<div style='border-left:4px solid {color};padding:6px 12px;margin:4px 0;"
                f"border-radius:4px;background:#f8fafc;'>"
                f"<b style='color:{color};'>Turn {t['turn']} · {t['speaker']}</b><br>{t['message']}</div>",
                unsafe_allow_html=True,
            )

        if st.button("🔍 Analyze Conversation", type="primary"):
            with st.spinner("Analyzing conversation…"):
                st.markdown("---")
                st.markdown("### 📊 Analysis Results")

                # ── ML per-turn
                ml_col, claude_col = st.columns(2)

                with ml_col:
                    st.markdown("#### 🔬 ML Model — Per-Turn Intents")
                    if ML_AVAILABLE:
                        history_ctx = []
                        ml_flow = []
                        for t in st.session_state.turns:
                            history_ctx.append(t["message"])
                            ctx = " [SEP] ".join(history_ctx)
                            label, conf = predict_ml(ctx)
                            ml_flow.append(label)
                            conf_str = f"{conf}%" if conf else "N/A"
                            st.markdown(
                                f"**T{t['turn']}** {intent_badge(label)} "
                                f"<small style='color:#6B7280;'>{conf_str}</small> "
                                f"— <i>{t['message'][:50]}…</i>",
                                unsafe_allow_html=True,
                            )
                        st.markdown(f"**Flow:** {' → '.join(ml_flow)}")
                        dominant_ml = max(set(ml_flow), key=ml_flow.count)
                        st.markdown(f"**Dominant:** {intent_badge(dominant_ml)}", unsafe_allow_html=True)
                        escalated_ml = any("escalation" in i.lower() for i in ml_flow)
                        st.markdown(f"**Escalation:** {'🚨 YES' if escalated_ml else '✅ No'}")
                    else:
                        st.warning("ML model not loaded.")

                # ── Claude API
                with claude_col:
                    st.markdown("#### 🤖 Gemini AI — Deep Analysis")
                    try:
                        result = analyze_with_claude_multiturn(st.session_state.turns)
                        for turn_res in result.get("turns", []):
                            badge = intent_badge(turn_res["intent"])
                            st.markdown(
                                f"**T{turn_res['turn']}** {badge} "
                                f"<small style='color:#6B7280;'>{turn_res['confidence']}</small> "
                                f"— <i>{turn_res['message_preview'][:50]}</i>",
                                unsafe_allow_html=True,
                            )
                        flow = result.get("intent_flow", [])
                        st.markdown(f"**Flow:** {' → '.join(flow)}")
                        st.markdown(f"**Dominant:** {intent_badge(result.get('dominant_intent',''))}", unsafe_allow_html=True)
                        esc = result.get("escalation_detected", False)
                        st.markdown(f"**Escalation:** {'🚨 YES' if esc else '✅ No'}")
                        st.markdown(f"**Overall Tone:** {result.get('overall_tone','')}")
                        st.info(f"📝 **Summary:** {result.get('conversation_summary','')}")
                        st.success(f"✅ **Recommended Action:** {result.get('recommended_action','')}")
                    except Exception as e:
                        st.error(f"Gemini API error: {e}")
    else:
        st.info("Add turns above or load the example conversation, then click Analyze.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption("🧠 Multi-Turn Chat Intent Classifier · Team 1 · Module 7 NLP · Powered by Google Gemini + Scikit-learn")
