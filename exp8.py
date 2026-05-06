import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ---------------- LOAD DATA ----------------
df = pd.read_csv("final_dataset.csv")

# ---------------- FEATURES ----------------
X_text = df['cleaned']
y = df['label']

# ---------------- TF-IDF ----------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(X_text)

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
mlp = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=300)
mlp.fit(X_train, y_train)

# ---------------- PREDICTION ----------------
y_pred = mlp.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt

plt.plot(mlp.loss_curve_)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Neural Network Loss Curve")
plt.show()