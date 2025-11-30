# app/streamlit_app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import joblib
import pandas as pd
from src.preprocess import clean_text
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="SMS Spam Classifier — Improved UI", layout="centered")

# --- Styling (small) ---
st.markdown("""
    <style>
    .result-card { padding: 16px; border-radius: 10px; color: #111; }
    .spam-bg { background: linear-gradient(90deg, #ffd6d6, #ffecec); border: 1px solid #ffb3b3; }
    .ham-bg { background: linear-gradient(90deg, #e6ffe6, #f0fff0); border: 1px solid #b6e0b6; }
    .big-label { font-size: 26px; font-weight:700; margin:0; }
    .small-muted { color: #666; font-size: 13px; }
    .conf-text { font-weight:600; }
    </style>
""", unsafe_allow_html=True)

st.title("📩 SMS Spam Classifier")
st.write("Type a message below and press **Predict**. The app shows confidence and helpful explainability information.")

# --- Load artifacts safely ---
def load_artifacts():
    model_path = os.path.join("models", "model.joblib")
    vect_path = os.path.join("models", "vectorizer.joblib")
    missing = []
    if not os.path.exists(model_path):
        missing.append(model_path)
    if not os.path.exists(vect_path):
        missing.append(vect_path)
    if missing:
        raise FileNotFoundError(f"Missing model/vectorizer files: {', '.join(missing)}")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vect_path)
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
    loaded_ok = True
except Exception as e:
    st.error("Model or vectorizer not found. Train the model first: `python src\\train.py`.")
    st.info(f"Error: {e}")
    loaded_ok = False

# --- Input area (centered) ---
st.markdown("### 🔎 Predict a message")
with st.form(key="predict_form"):
    user_text = st.text_area("Enter SMS message", height=140, placeholder="Type or paste a message here...")
    col_btn1, col_btn2 = st.columns([1,1])
    with col_btn1:
        predict_btn = st.form_submit_button("Predict")
    with col_btn2:
        clear_btn = st.form_submit_button("Clear")
    # form prevents accidental double clicks

if clear_btn:
    # simply refresh page to clear
    st.rerun()

if predict_btn:
    if not user_text.strip():
        st.warning("Please enter a message first.")
    elif not loaded_ok:
        st.error("Model not available. Train model and try again.")
    else:
        # Preprocess and vectorize
        cleaned = clean_text(user_text)
        x = vectorizer.transform([cleaned])

        # Predict and probabilities (if supported)
        try:
            probs = model.predict_proba(x)[0]
            # map probabilities to class labels in order
            classes = list(model.classes_)
            # create mapping label -> probability
            prob_map = {str(classes[i]): float(probs[i]) for i in range(len(classes))}
            # predicted class
            pred = model.predict(x)[0]
            confidence = prob_map[str(pred)]
        except Exception:
            # model doesn't support predict_proba (unlikely for MultinomialNB), fallback
            pred = model.predict(x)[0]
            confidence = 1.0
            prob_map = {str(pred): 1.0}

        # Format confidence
        pct = f"{confidence*100:.1f}%"

        # Color-coded card
        if str(pred).lower() == "spam":
            card_html = f"<div class='result-card spam-bg'><p class='big-label'>🚨 SPAM</p><p class='conf-text'>Confidence: {pct}</p></div>"
        else:
            card_html = f"<div class='result-card ham-bg'><p class='big-label'>✔️ HAM (Not spam)</p><p class='conf-text'>Confidence: {pct}</p></div>"

        st.markdown(card_html, unsafe_allow_html=True)

        # Visual confidence bar
        st.progress(int(confidence*100))

        # show probability breakdown (if available)
        if len(prob_map) > 1:
            st.subheader("Probability breakdown")
            prob_df = pd.DataFrame(list(prob_map.items()), columns=["label", "probability"])
            prob_df["probability_pct"] = (prob_df["probability"] * 100).round(1).astype(str) + '%'
            # display sorted
            prob_df = prob_df.sort_values("probability", ascending=False).reset_index(drop=True)
            st.table(prob_df[["label", "probability_pct"]])

        # Top tokens that indicate spam (explainability)
        try:
            classes = list(model.classes_)
            if "spam" in classes:
                spam_idx = classes.index("spam")
                probs_vec = model.feature_log_prob_[spam_idx]
                feat_names = vectorizer.get_feature_names_out()
                top_idx = probs_vec.argsort()[-12:][::-1]
                top_tokens = [feat_names[i] for i in top_idx]
                st.subheader("Top tokens that indicate spam")
                st.write(", ".join(top_tokens))
        except Exception:
            # non-fatal
            pass

# ===== Dataset preview below prediction =====
st.markdown("---")
st.header("📁 Dataset Preview & Class Distribution")
default_path = os.path.join("data", "spam.csv")
st.caption("App reads the CSV at data/spam.csv (make sure it's present and well-formatted).")

if os.path.exists(default_path):
    try:
        df = pd.read_csv(default_path, encoding="latin-1", header=0, engine="python", usecols=[0,1])
        if df.shape[1] > 1:
            df.columns = ["label", "text"]
        if st.button("Show dataset sample"):
            st.dataframe(df.sample(min(10, len(df))).reset_index(drop=True))
        st.subheader("Class distribution")
        st.bar_chart(df["label"].value_counts())
    except Exception as e:
        st.error(f"Could not read dataset: {e}")
        st.write("If your messages include commas, ensure the text field is quoted.")
else:
    st.info("No dataset found at data/spam.csv. You can upload using the uploader below.")

uploaded = st.file_uploader("Upload CSV to preview (optional)", type=["csv"])
if uploaded is not None:
    try:
        udf = pd.read_csv(uploaded, encoding="latin-1", header=0, engine="python")
        if udf.shape[1] > 1:
            udf.columns = ["label", "text"]
        st.write("Preview of uploaded file:")
        st.dataframe(udf.head(10))
        st.write("Class distribution:")
        st.bar_chart(udf['label'].value_counts())
    except Exception as e:
        st.error(f"Couldn't read uploaded file: {e}")

# =======================
# FOOTER (name + links)
# =======================
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 15px;'>
<strong>SMS Spam Classifier Project</strong><br><br>
Developed by <strong>AJAYA RAJ A N</strong><br>
<a href='https://www.linkedin.com/in/ajayaraj98' target='_blank'>LinkedIn</a> |
<a href='https://github.com/Ajay-araj' target='_blank'>GitHub</a>
</div>
""", unsafe_allow_html=True)

st.caption("Tip: If Streamlit asks for an email on first use, skip it — the app runs locally without signing in.")
