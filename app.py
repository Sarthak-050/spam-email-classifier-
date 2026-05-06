import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.title("📧 Advanced Spam Email Analyzer")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("final_dataset.csv")

df = load_data()

# ---------------- TRAIN MODEL ----------------
@st.cache_resource 
def train_model(df):
    X_text = df['cleaned']
    y = df['label']

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vectorizer.fit_transform(X_text)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return vectorizer, model

vectorizer, model = train_model(df)

# ---------------- HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- INPUT ----------------
user_input = st.text_area("Enter Email")

# ---------------- KEYWORD RULES ----------------
fraud_keywords = ["lottery", "win", "prize", "urgent", "bank", "otp"]
promo_keywords = ["offer", "discount", "sale", "buy", "install", "free"]

def analyze_reason(text):
    text = text.lower()
    reasons = []

    for word in fraud_keywords:
        if word in text:
            reasons.append(f"Fraud keyword detected: {word}")

    for word in promo_keywords:
        if word in text:
            reasons.append(f"Promotional keyword detected: {word}")

    return reasons

def detect_type(text):
    text = text.lower()
    if any(word in text for word in fraud_keywords):
        return "⚠️ Fraud Spam"
    elif any(word in text for word in promo_keywords):
        return "📢 Promotional Spam"
    else:
        return "🟡 General Spam"

# ---------------- PREDICTION ----------------
if st.button("Analyze Email"):

    if user_input.strip() == "":
        st.warning("Enter a message")
    else:
        cleaned = re.sub(r'[^a-zA-Z]', ' ', user_input.lower())
        vec = vectorizer.transform([cleaned])

        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][1]

        # ---------------- OUTPUT ----------------
        if pred == 1:
            st.error("🚫 SPAM")
            spam_type = detect_type(user_input)
            st.write("Type:", spam_type)
        else:
            st.success("✅ NOT SPAM")
            spam_type = "Not Spam"

        st.write(f"📊 Spam Probability: {prob*100:.2f}%")

        # ---------------- REASONS ----------------
        reasons = analyze_reason(user_input)

        if reasons:
            st.subheader("🔍 Why this email is spam?")
            for r in reasons:
                st.write("-", r)
        else:
            st.write("No strong spam indicators found")

        # ---------------- SAVE HISTORY ----------------
        st.session_state.history.append({
            "text": user_input,
            "prediction": "Spam" if pred == 1 else "Not Spam",
            "probability": prob
        })

# ---------------- HISTORY SECTION ----------------
st.subheader("📊 History Analysis")

if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)

    st.dataframe(hist_df)

    # Chart
    counts = hist_df['prediction'].value_counts()
    st.bar_chart(counts)

    # Common words in spam
    spam_text = " ".join(hist_df[hist_df['prediction']=="Spam"]['text'])

    words = spam_text.split()
    common_words = pd.Series(words).value_counts().head(10)

    st.subheader("🔥 Common Words in Spam")
    st.write(common_words)