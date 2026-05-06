# Experiment 2: Checking Linear Separability (FINAL FIXED)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ---------------- LOAD FINAL DATA ----------------

df = pd.read_csv("final_dataset.csv")

print("Columns:", df.columns)

# ---------------- FEATURES ----------------

X_text = df['Message']   # already exists
y = df['label']          # already numeric

# ---------------- TF-IDF ----------------

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X = vectorizer.fit_transform(X_text)

# ---------------- PCA ----------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# ---------------- PLOT ----------------

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Spam vs Ham Distribution")
plt.show()