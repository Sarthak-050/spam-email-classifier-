import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report

# ---------------- LOAD DATA ----------------
df = pd.read_csv("final_dataset.csv")

# ---------------- FEATURES ----------------
X_text = df['cleaned']
y = df['label']

# ---------------- TF-IDF ----------------
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X = vectorizer.fit_transform(X_text)

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = Perceptron()
model.fit(X_train, y_train)

# ---------------- PREDICTION ----------------
y_pred = model.predict(X_test)

# ---------------- EVALUATION ----------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
