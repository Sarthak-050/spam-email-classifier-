# Experiment 6: Final Spam Detection with User Input

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------- LOAD DATA ----------------
df = pd.read_csv("final_dataset.csv")

# ---------------- FEATURES ----------------
X_text = df['cleaned']
y = df['label']

# ---------------- TF-IDF ----------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(X_text)

# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ----------- USER INPUT LOOP -----------

while True:
    user_input = input("\nEnter email text (type 'exit' to stop):\n")

    if user_input.lower() == "exit":
        break

    # clean input same as training
    import re
    cleaned = re.sub(r'[^a-zA-Z]', ' ', user_input.lower())

    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    if pred == 1:
        print("🚫 SPAM")
    else:
        print("✅ NOT SPAM")