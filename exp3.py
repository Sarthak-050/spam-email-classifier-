import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

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

# =====================================================
# METHOD 1: CLOSED FORM
# =====================================================

X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

X_train_bias = np.c_[np.ones(X_train_dense.shape[0]), X_train_dense]
X_test_bias = np.c_[np.ones(X_test_dense.shape[0]), X_test_dense]

# Compute theta using pseudo inverse

theta = np.linalg.pinv(X_train_bias) @ y_train

# Prediction

y_pred = X_test_bias @ theta

# Convert to classification

y_pred_class = (y_pred >= 0.5).astype(int)

print("Closed Form Accuracy:", accuracy_score(y_test, y_pred_class))
print("Closed Form MSE:", mean_squared_error(y_test, y_pred))

# =====================================================
# METHOD 2: SGD REGRESSOR
# =====================================================

sgd = SGDRegressor(max_iter=1000, tol=1e-3)
sgd.fit(X_train, y_train)

pred_sgd = sgd.predict(X_test)
pred_sgd_class = (pred_sgd >= 0.5).astype(int)

print("\nSGD Accuracy:", accuracy_score(y_test, pred_sgd_class))
print("SGD MSE:", mean_squared_error(y_test, pred_sgd))

# =====================================================
# METHOD 3: SKLEARN LINEAR REGRESSION
# =====================================================

lr = LinearRegression()
lr.fit(X_train_dense, y_train)

pred_lr = lr.predict(X_test_dense)
pred_lr_class = (pred_lr >= 0.5).astype(int)

print("\nLinear Regression Accuracy:", accuracy_score(y_test, pred_lr_class))
print("Linear Regression MSE:", mean_squared_error(y_test, pred_lr))
