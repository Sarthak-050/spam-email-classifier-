# Experiment 4: Linear Regression for Spam Detection (FINAL)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error

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

# ---------------- METHOD 1: Linear Regression ----------------

lr = LinearRegression()
lr.fit(X_train.toarray(), y_train)

pred_lr = lr.predict(X_test.toarray())

print("Linear Regression MSE:", mean_squared_error(y_test, pred_lr))

# ---------------- METHOD 2: SGD Regressor ----------------

sgd = SGDRegressor(max_iter=1000, tol=1e-3)
sgd.fit(X_train, y_train)

pred_sgd = sgd.predict(X_test)

print("SGD Regressor MSE:", mean_squared_error(y_test, pred_sgd))