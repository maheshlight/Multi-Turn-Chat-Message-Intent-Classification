"""
app.py — Multi-Turn Chat Intent Classification
Team 1 | Module 7 NLP Project
Pure ML version — no API key required
"""

import os
import re
import joblib
import numpy as np
import streamlit as st

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

# ─────────────────────────────────────────────────────────────────────────────
# Keyword-based intent classifier (no API needed)
# ─────────────────────────────────────────────────────────────────────────────
INTENT_KEYWORDS = {
    "Greeting":      ["hello","hi","hey","good morning","good afternoon","good evening",
                      "howdy","greetings","what's up","how are you","nice to meet"],
    "Farewell":      ["bye","goodbye","see you","take care","farewell","good night",
                      "talk later","have a good","that's all i needed","thanks bye"],
    "Appreciation":  ["thank","thanks","appreciate","grateful","awesome","amazing",
                      "excellent","brilliant","wonderful","fantastic","great job",
                      "well done","you've been","so helpful","really helped"],
    "Complaint":     ["frustrated","angry","terrible","worst","awful","unacceptable",
                      "useless","broken","horrible","fed up","disappointed","furious",
                      "disgusted","ridiculous","not working","still not","wrong","damaged",
                      "issue","problem","complaint","unhappy","dissatisfied","failed"],
    "Escalation":    ["manager","supervisor","escalate","higher authority","not acceptable",
                      "legal","sue","report","demand","right now","immediately fix",
                      "speak to someone","get me your","i want to speak"],
    "Request":       ["please","can you","could you","i need","i want","i'd like",
                      "would you","sign me up","cancel","refund","update","change",
                      "send me","book","schedule","order","purchase","subscribe"],
    "Query":         ["what","how","when","where","why","which","who","is there",
                      "do you","does","can i","available","tell me","explain",
                      "information","details","know about","find out","policy"],
    "Feedback":      ["suggest","suggestion","improve","improvement","better","idea",
                      "should add","you should","consider","feature","option","wish",
                      "would be nice","recommend adding","feedback","opinion"],
    "Clarification": ["clarify","unclear","confused","don't understand","what do you mean",
                      "explain","not sure","elaborate","mean by","what is meant",
                      "can you repeat","didn't get","please explain"],
}

def classify_intent(text: str) -> tuple:
    """Classify intent using keyword matching with scoring."""
    t = text.lower()
    scores = {intent: 0 for intent in INTENT_KEYWORDS}
    matched = {intent: [] for intent in INTENT_KEYWORDS}

    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                scores[intent] += 1
                matched[intent].append(kw)

    # Get top intents
    sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_intents[0][0] if sorted_intents[0][1] > 0 else "Query"
    primary_score = sorted_intents[0][1]

    # Secondary intents (score > 0 and not primary)
    secondary = [i for i, s in sorted_intents[1:] if s > 0][:2]

    # Confidence based on score
    if primary_score >= 3:   conf = "High"
    elif primary_score >= 2: conf = "Medium"
    elif primary_score >= 1: conf = "Medium"
    else:                    conf = "Low"

    # Key signals
    signals = matched[primary][:4] or ["(no strong signals)"]

    return primary, secondary, conf, signals

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
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def intent_badge(intent: str) -> str:
    color = INTENT_COLORS.get(intent, "#6B7280")
    emoji = INTENT_EMOJI.get(intent, "🔵")
    return (f'<span style="background:{color};color:#fff;padding:3px 10px;'
            f'border-radius:12px;font-weight:600;font-size:0.85rem;">'
            f'{emoji} {intent}</span>')

def confidence_bar(conf: str) -> str:
    pct = {"High": 90, "Medium": 60, "Low": 30}.get(conf, 50)
    color = "#10B981" if pct >= 75 else "#F59E0B" if pct >= 50 else "#EF4444"
    return (f'<div style="background:#e5e7eb;border-radius:6px;height:10px;width:100%;">'
            f'<div style="background:{color};border-radius:6px;height:10px;width:{pct}%;"></div></div>'
            f'<small style="color:{color};font-weight:600;">{conf} ({pct}%)</small>')

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

    st.markdown("**Intent Categories**")
    for label in INTENT_LABELS:
        color = INTENT_COLORS[label]
        emoji = INTENT_EMOJI[label]
        st.markdown(
            f"<span style='color:{color};font-weight:600;'>{emoji} {label}</span>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.success("✅ Keyword ML Engine Ready")
    st.info("💡 No API key needed!\nPure keyword-based classification.")

# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────
st.title("💬 Multi-Turn Chat Intent Classifier")
st.caption("Classify chat messages · Powered by Keyword ML Engine · No API key required")
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
            tone    = emotional_tone(user_text)
            urgency = urgency_level(user_text)
            signals = key_signals(user_text)
            primary, secondary, conf, intent_signals = classify_intent(user_text)

            st.markdown("---")
            st.markdown("### 📊 Analysis Results")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("**🎯 Primary Intent**")
                st.markdown(intent_badge(primary), unsafe_allow_html=True)
                st.markdown(confidence_bar(conf), unsafe_allow_html=True)
            with col_b:
                st.markdown("**😊 Emotional Tone**")
                st.markdown(f"### {tone}")
            with col_c:
                st.markdown("**⚡ Urgency Level**")
                st.markdown(f"### {urgency}")

            st.markdown("---")

            col_d, col_e = st.columns(2)
            with col_d:
                if secondary:
                    st.markdown("**🔀 Secondary Intents**")
                    badges = " ".join(intent_badge(s) for s in secondary)
                    st.markdown(badges, unsafe_allow_html=True)
            with col_e:
                st.markdown("**🔑 Key Signals Detected**")
                all_sig = list(dict.fromkeys(intent_signals + signals))[:6]
                if all_sig and all_sig != ["(no strong signals)"]:
                    st.markdown(" ".join(f"`{s}`" for s in all_sig))
                else:
                    st.markdown("`(no strong signals)`")

            # Explanation
            st.markdown("---")
            explanation = f"The text was classified as **{primary}** based on keyword patterns. "
            if intent_signals and intent_signals != ["(no strong signals)"]:
                explanation += f"Key words detected: *{', '.join(intent_signals)}*. "
            if secondary:
                explanation += f"Secondary intents detected: *{', '.join(secondary)}*."
            st.info(f"💬 {explanation}")

    elif analyze_btn:
        st.warning("Please enter some text to analyze.")

# ─── Multi-Turn Mode ─────────────────────────────────────────────────────────
else:
    st.subheader("🔄 Multi-Turn Conversation Analyzer")

    if "turns" not in st.session_state:
        st.session_state.turns = []
    if "turn_counter" not in st.session_state:
        st.session_state.turn_counter = 1

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

    if st.session_state.turns:
        st.markdown("#### 💬 Conversation")
        for t in st.session_state.turns:
            color = "#3B82F6" if t["speaker"] == "User" else "#10B981"
            st.markdown(
                f"<div style='border-left:4px solid {color};padding:6px 12px;margin:4px 0;"
                f"border-radius:4px;background:#f8fafc;'>"
                f"<b style='color:{color};'>Turn {t['turn']} · {t['speaker']}</b>"
                f"<br>{t['message']}</div>",
                unsafe_allow_html=True,
            )

        if st.button("🔍 Analyze Conversation", type="primary"):
            with st.spinner("Analyzing conversation…"):
                st.markdown("---")
                st.markdown("### 📊 Analysis Results")

                intent_flow = []
                turn_results = []

                for t in st.session_state.turns:
                    primary, secondary, conf, sig = classify_intent(t["message"])
                    tone = emotional_tone(t["message"])
                    intent_flow.append(primary)
                    turn_results.append({
                        "turn": t["turn"],
                        "speaker": t["speaker"],
                        "message": t["message"],
                        "intent": primary,
                        "confidence": conf,
                        "tone": tone,
                        "signals": sig,
                    })

                # Per-turn results
                st.markdown("#### 🔬 Per-Turn Intent Analysis")
                for r in turn_results:
                    badge = intent_badge(r["intent"])
                    st.markdown(
                        f"**T{r['turn']}** {badge} "
                        f"<small style='color:#6B7280;'>{r['confidence']}</small> "
                        f"— <i>{r['message'][:60]}…</i>",
                        unsafe_allow_html=True,
                    )

                st.markdown("---")

                # Summary stats
                col1, col2, col3, col4 = st.columns(4)

                dominant = max(set(intent_flow), key=intent_flow.count)
                escalated = any("Escalation" in i for i in intent_flow)
                neg_count = sum(1 for r in turn_results if "Negative" in r["tone"])
                pos_count = sum(1 for r in turn_results if "Positive" in r["tone"])
                overall_tone = "😠 Negative" if neg_count > pos_count else "😊 Positive" if pos_count > 0 else "😶 Neutral"

                with col1:
                    st.markdown("**🏆 Dominant Intent**")
                    st.markdown(intent_badge(dominant), unsafe_allow_html=True)
                with col2:
                    st.markdown("**🚨 Escalation**")
                    st.markdown("### 🚨 YES" if escalated else "### ✅ No")
                with col3:
                    st.markdown("**😊 Overall Tone**")
                    st.markdown(f"### {overall_tone}")
                with col4:
                    st.markdown("**📊 Total Turns**")
                    st.markdown(f"### {len(intent_flow)}")

                st.markdown("---")
                st.markdown(f"**🔀 Intent Flow:** {' → '.join(intent_badge(i) for i in intent_flow)}", unsafe_allow_html=True)

                # Conversation summary
                st.markdown("---")
                unique_intents = list(dict.fromkeys(intent_flow))
                summary = (f"The conversation spans {len(intent_flow)} turns covering: "
                           f"{', '.join(unique_intents)}. "
                           f"{'⚠️ Escalation was detected — immediate attention required.' if escalated else '✅ No escalation detected.'}")
                st.info(f"📝 **Summary:** {summary}")

                if escalated:
                    st.error("✅ **Recommended Action:** Escalate to senior support immediately.")
                elif "Complaint" in intent_flow:
                    st.warning("✅ **Recommended Action:** Address complaint with priority response.")
                else:
                    st.success("✅ **Recommended Action:** Standard response — follow up as needed.")
    else:
        st.info("Add turns above or load the example conversation, then click Analyze.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption("🧠 Multi-Turn Chat Intent Classifier · Team 1 · Module 7 NLP · Keyword ML Engine · No API Key Required")
